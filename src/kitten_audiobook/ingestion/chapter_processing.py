from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .chapter_detection import ChapterDetector, ChapterInfo, detect_chapters_enhanced
from .structure import Document
from ..text.chunking import Chunk
from ..tts.engine import TTSEngine
from ..tts.synthesis import synthesize_chunks, assemble_with_pauses, PausePlan
from ..audio.assembly import save_wav
from ..audio.post import loudness_normalize, encode_mp3
from ..utils.io import ensure_dirs


@dataclass
class ChapterAudio:
    """Contains audio and metadata for a single chapter"""
    chapter_info: ChapterInfo
    audio: np.ndarray
    chunks: List[Chunk]
    duration_seconds: float
    file_path: Optional[Path] = None
    mp3_path: Optional[Path] = None
    

@dataclass
class ChapterProcessingResult:
    """Result of chapter-by-chapter processing"""
    chapters: List[ChapterAudio]
    full_audio: Optional[np.ndarray] = None
    manifest: Optional[Dict[str, Any]] = None
    zip_path: Optional[Path] = None
    

class ChapterProcessor:
    """Handles chapter-by-chapter audiobook processing"""
    
    def __init__(
        self,
        document: Document,
        pdf_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        voice: str = "expr-voice-4-f",
        speed: float = 1.0,
        normalize_loudness: bool = True
    ):
        self.document = document
        self.pdf_path = pdf_path
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="chapters_"))
        self.voice = voice
        self.speed = speed
        self.normalize_loudness = normalize_loudness
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect chapters
        self.chapters_info, self.detector = detect_chapters_enhanced(document, pdf_path)
        logging.info(f"Detected {len(self.chapters_info)} chapters")
        
    def process_chapters_incrementally(
        self,
        engine: TTSEngine,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ChapterProcessingResult:
        """
        Process chapters incrementally, saving each chapter as it completes
        
        Args:
            engine: TTS engine to use
            chunks: Text chunks to process
            progress_callback: Optional callback for progress updates (current_chapter, total_chapters, status)
            
        Returns:
            ChapterProcessingResult with all chapter audio and metadata
        """
        if not self.chapters_info:
            # Fallback to single chapter processing
            return self._process_as_single_chapter(engine, chunks, progress_callback)
        
        # Get chapter boundaries for chunks
        chapter_boundaries = self.detector.get_chapter_boundaries_for_chunks(chunks)
        chapter_ranges = self._compute_chapter_ranges(chapter_boundaries, len(chunks))
        
        logging.info(f"Processing {len(chapter_ranges)} chapters incrementally")
        
        chapter_audios = []
        all_audio_segments = []
        
        for i, (start_idx, end_idx) in enumerate(chapter_ranges):
            chapter_info = self.chapters_info[i] if i < len(self.chapters_info) else ChapterInfo(
                title=f"Chapter {i + 1}",
                start_page=1,
                source="generated"
            )
            
            if progress_callback:
                progress_callback(i + 1, len(chapter_ranges), f"Processing: {chapter_info.title}")
            
            # Get chunks for this chapter
            chapter_chunks = chunks[start_idx:end_idx + 1]
            
            logging.info(f"Processing chapter {i + 1}/{len(chapter_ranges)}: '{chapter_info.title}' ({len(chapter_chunks)} chunks)")
            
            # Synthesize chapter audio
            try:
                chapter_audio = self._process_single_chapter(
                    engine, chapter_info, chapter_chunks, i + 1
                )
                chapter_audios.append(chapter_audio)
                all_audio_segments.append(chapter_audio.audio)
                
                logging.info(f"Chapter {i + 1} completed: {chapter_audio.duration_seconds:.1f}s")
                
            except Exception as e:
                logging.error(f"Failed to process chapter {i + 1}: {e}")
                # Create silence for failed chapter
                silence = np.zeros(int(24000 * 2), dtype=np.float32)
                chapter_audio = ChapterAudio(
                    chapter_info=chapter_info,
                    audio=silence,
                    chunks=chapter_chunks,
                    duration_seconds=2.0
                )
                chapter_audios.append(chapter_audio)
                all_audio_segments.append(silence)
        
        # Create combined audio with chapter pauses
        full_audio = self._combine_chapter_audios(all_audio_segments)
        
        # Create manifest
        manifest = self._create_manifest(chapter_audios)
        
        # Create ZIP file with all chapters
        zip_path = self._create_chapters_zip(chapter_audios)
        
        return ChapterProcessingResult(
            chapters=chapter_audios,
            full_audio=full_audio,
            manifest=manifest,
            zip_path=zip_path
        )
    
    def _process_single_chapter(
        self,
        engine: TTSEngine,
        chapter_info: ChapterInfo,
        chunks: List[Chunk],
        chapter_num: int
    ) -> ChapterAudio:
        """Process a single chapter"""
        # Synthesize all chunks for this chapter
        chunk_audios = []
        
        for chunk in tqdm(chunks, desc=f"Ch.{chapter_num}", leave=False):
            try:
                audio = engine.synthesize(chunk.text, voice=self.voice, speed=self.speed)
                chunk_audios.append(audio)
            except Exception as e:
                logging.warning(f"Failed to synthesize chunk in chapter {chapter_num}: {e}")
                # Add silence as fallback
                silence = np.zeros(int(24000 * 1), dtype=np.float32)
                chunk_audios.append(silence)
        
        # Combine chunks with small pauses
        if chunk_audios:
            pause_plan = PausePlan(chunk_pause_s=0.3)
            chapter_audio = assemble_with_pauses(chunk_audios, chunks, pause_plan)
        else:
            chapter_audio = np.zeros(0, dtype=np.float32)
        
        # Normalize loudness if requested
        if self.normalize_loudness and len(chapter_audio) > 0:
            chapter_audio = loudness_normalize(chapter_audio, 24000, target_lufs=-18.0)
        
        # Calculate duration
        duration = len(chapter_audio) / 24000
        
        # Save chapter audio files
        chapter_filename = f"chapter_{chapter_num:02d}_{self._sanitize_filename(chapter_info.title)}"
        wav_path = self.output_dir / f"{chapter_filename}.wav"
        mp3_path = self.output_dir / f"{chapter_filename}.mp3"
        
        save_wav(wav_path, chapter_audio, 24000)
        encode_mp3(wav_path, mp3_path, bitrate="80k")
        
        return ChapterAudio(
            chapter_info=chapter_info,
            audio=chapter_audio,
            chunks=chunks,
            duration_seconds=duration,
            file_path=wav_path,
            mp3_path=mp3_path
        )
    
    def _process_as_single_chapter(
        self,
        engine: TTSEngine,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ChapterProcessingResult:
        """Fallback: process as single chapter when no chapters detected"""
        if progress_callback:
            progress_callback(1, 1, "Processing as single chapter")
        
        chapter_info = ChapterInfo(
            title=self.document.title or "Audiobook",
            start_page=1,
            end_page=len(self.document.pages),
            source="fallback"
        )
        
        chapter_audio = self._process_single_chapter(engine, chapter_info, chunks, 1)
        
        manifest = self._create_manifest([chapter_audio])
        zip_path = self._create_chapters_zip([chapter_audio])
        
        return ChapterProcessingResult(
            chapters=[chapter_audio],
            full_audio=chapter_audio.audio,
            manifest=manifest,
            zip_path=zip_path
        )
    
    def _compute_chapter_ranges(self, boundaries: List[bool], total_chunks: int) -> List[Tuple[int, int]]:
        """Convert boundary flags to chapter ranges"""
        ranges = []
        start = 0
        
        for i in range(total_chunks):
            if i < len(boundaries) and boundaries[i]:
                # End current chapter at i
                ranges.append((start, i))
                start = i + 1
        
        # Add final chapter
        if start < total_chunks:
            ranges.append((start, total_chunks - 1))
        
        return ranges
    
    def _combine_chapter_audios(self, chapter_audios: List[np.ndarray]) -> np.ndarray:
        """Combine chapter audios with pauses between chapters"""
        if not chapter_audios:
            return np.zeros(0, dtype=np.float32)
        
        combined_parts = []
        chapter_pause_samples = int(24000 * 1.5)  # 1.5 second pause between chapters
        
        for i, audio in enumerate(chapter_audios):
            combined_parts.append(audio.astype(np.float32))
            
            # Add pause between chapters (except after last chapter)
            if i < len(chapter_audios) - 1:
                pause = np.zeros(chapter_pause_samples, dtype=np.float32)
                combined_parts.append(pause)
        
        return np.concatenate(combined_parts)
    
    def _create_manifest(self, chapter_audios: List[ChapterAudio]) -> Dict[str, Any]:
        """Create chapter manifest/playlist"""
        manifest = {
            "title": self.document.title or "Audiobook",
            "author": self.document.author,
            "voice": self.voice,
            "speed": self.speed,
            "total_chapters": len(chapter_audios),
            "total_duration": sum(ch.duration_seconds for ch in chapter_audios),
            "chapters": []
        }
        
        cumulative_time = 0.0
        for i, chapter_audio in enumerate(chapter_audios):
            chapter_data = {
                "index": i + 1,
                "title": chapter_audio.chapter_info.title,
                "duration": chapter_audio.duration_seconds,
                "start_time": cumulative_time,
                "end_time": cumulative_time + chapter_audio.duration_seconds,
                "start_page": chapter_audio.chapter_info.start_page,
                "end_page": chapter_audio.chapter_info.end_page,
                "chunks_count": len(chapter_audio.chunks),
                "file_name": chapter_audio.mp3_path.name if chapter_audio.mp3_path else None
            }
            manifest["chapters"].append(chapter_data)
            cumulative_time += chapter_audio.duration_seconds + 1.5  # Add chapter pause
        
        return manifest
    
    def _create_chapters_zip(self, chapter_audios: List[ChapterAudio]) -> Path:
        """Create ZIP file containing all chapter audio files"""
        zip_path = self.output_dir / "chapters.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add chapter audio files
            for chapter_audio in chapter_audios:
                if chapter_audio.mp3_path and chapter_audio.mp3_path.exists():
                    zf.write(chapter_audio.mp3_path, chapter_audio.mp3_path.name)
                if chapter_audio.file_path and chapter_audio.file_path.exists():
                    zf.write(chapter_audio.file_path, chapter_audio.file_path.name)
            
            # Add manifest
            manifest = self._create_manifest(chapter_audios)
            manifest_json = json.dumps(manifest, indent=2)
            zf.writestr("manifest.json", manifest_json)
        
        return zip_path
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem compatibility"""
        # Remove or replace problematic characters
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'\s+', '_', sanitized)
        return sanitized[:50]  # Limit length
    
    def get_chapter_info(self) -> List[Dict[str, Any]]:
        """Get chapter information for UI display"""
        return [
            {
                "title": ch.title,
                "start_page": ch.start_page,
                "end_page": ch.end_page,
                "level": ch.level,
                "source": ch.source
            }
            for ch in self.chapters_info
        ]
