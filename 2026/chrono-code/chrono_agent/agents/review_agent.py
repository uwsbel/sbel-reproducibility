"""
Review Agent (Agent 3) - VLM Visual Description Only.

This agent describes simulation camera images from cam/.
It does not decide correctness, physics validity, or code fixes; downstream
physics analysis interprets the descriptions together with CSV output.
"""

import asyncio
import base64
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from chrono_agent.agents.base import BaseAgent, _diff_usage
from chrono_agent.agents.exceptions import AgentLLMError
from chrono_agent.workflow.events import emit_agent_lifecycle_event
from chrono_agent.agents.prompts import review_prompts
from chrono_agent.models.plan import SimulationPlan
from chrono_agent.models.review import ReviewResult
from chrono_agent.config import get_settings

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")
# Override Gemini's default 1-fps sampling for inline video. Sub-2 s
# clips would otherwise resolve to a single frame and lose all motion
# signal (session_20260429_161918 missed visible platform fall).
# API range [0, 24]; per-frame cost ~263 tokens.
GEMINI_VIDEO_FPS = 5.0

_MIME_MAP_IMAGE = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
}

_MIME_MAP_VIDEO = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


class ReviewAgent(BaseAgent):
    """
    Agent 3: VLM Visual Description Only.

    Reads camera images from cam/, describes each one individually,
    and passes descriptions to the downstream physics analysis stage.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        super().__init__(
            agent_name="ReviewAgent",
            agent_number=3,
            llm_provider=llm_provider,
            model=model,
            temperature=temperature,
        )

    async def execute(
        self,
        plan: SimulationPlan,
        execution_result: dict,
        generated_code: Optional[str] = None,
    ) -> ReviewResult:
        """Execute VLM visual description (required by BaseAgent abstract method)."""
        return await self.review_visual_output(
            plan,
            execution_result,
            generated_code=generated_code,
        )

    async def review_visual_output(
        self,
        plan: SimulationPlan,
        execution_result: dict,
        generated_code: Optional[str] = None,
    ) -> ReviewResult:
        """Describe simulation camera images from cam/.

        Scans the cam/ directory for images (or video frames sampled from mp4),
        and returns the combined descriptions for downstream physics analysis.
        """
        self.logger.info("Describing camera images with VLM...")

        # Collect images from cam/
        settings = get_settings()
        cams_dir = Path(settings.visualization_output_path) / "cam"
        images = self._collect_cam_images(cams_dir)

        if not images:
            if self._expects_camera_images(plan):
                description = (
                    "No camera images found in cam/ but visualization mode "
                    "expects sensor camera output. Camera setup may have failed or "
                    "images were written to an unexpected directory."
                )
                return ReviewResult(
                    approved=False,
                    physics_check="invalid",
                    common_sense_check="invalid",
                    feedback=description,
                    visual_quality_score=0,
                    visual_issues=["No camera images produced despite sensor_camera mode"],
                    analyzed_frames=[],
                    video_description=description,
                )
            description = "No camera images found (not expected in this visualization mode)."
            return ReviewResult(
                approved=True,
                physics_check="valid",
                common_sense_check="valid",
                feedback=description,
                visual_quality_score=None,
                visual_issues=[],
                analyzed_frames=[],
                video_description=description,
            )

        # Build context
        objectives = [obj for obj in plan.objectives] if plan.objectives else []
        plan_summary = self._build_plan_summary(plan)

        # Describe each image individually
        descriptions: List[str] = []
        for image_path in images:
            label = Path(image_path).stem
            try:
                desc = await self._describe_single_image(image_path, label, objectives, plan_summary)
                descriptions.append(f"[{label}]\n{desc}")
            except AgentLLMError:
                raise
            except Exception as e:
                self.logger.error(f"Failed to describe {image_path}: {e}")
                descriptions.append(f"[{label}]\nFailed to describe: {e}")

        combined_description = "\n\n".join(descriptions)
        self.logger.info(f"VLM descriptions complete for {len(images)} images")

        return ReviewResult(
            approved=True,
            physics_check="valid",
            common_sense_check="valid",
            feedback=combined_description,
            visual_quality_score=None,
            visual_issues=[],
            analyzed_frames=images,
            video_description=combined_description,
        )

    # ------------------------------------------------------------------
    # Helper: provider-aware multimodal message building
    # ------------------------------------------------------------------

    async def _call_vlm(
        self,
        text_prompt: str,
        image_paths: Optional[List[str]] = None,
        operation: str = "vlm_call",
    ) -> str:
        """Send a text (+optional images) request via the provider SDK and return the text response.

        Persists prompt + response to ``dialog_manager`` so debugging a review
        FAIL does not require re-running the pipeline. When the provider
        emits chain-of-thought / reasoning_content alongside the visible
        answer (Claude extended-thinking, Gemini 2.5 thought summaries,
        DeepSeek-R1 / OpenAI o1+ reasoning_content), it is captured into
        a ``# THINKING`` section of the dialog response file so a failed
        review can be diagnosed against what the model was actually
        reasoning about, not just its terse JSON verdict. Raises
        ``AgentLLMError`` on any SDK / network failure.
        """
        metadata = {
            "operation": operation,
            "provider": self.provider,
            "model": self.model,
            "num_images": len(image_paths) if image_paths else 0,
            "image_paths": list(image_paths) if image_paths else [],
        }

        if self.dialog_manager is not None:
            try:
                await asyncio.to_thread(
                    self.dialog_manager.log_prompt,
                    self.agent_name,
                    text_prompt,
                    metadata,
                )
            except Exception as exc:
                self.logger.warning(f"Failed to log VLM prompt ({operation}): {exc}")

        # Pipeline-stats lifecycle wrap: ReviewAgent bypasses
        # ``BaseAgent.invoke_llm``, so without an explicit started/finished
        # pair here the engine's _PipelineStatsCollector never sees this
        # agent and the panel under-reports tokens. The per-provider
        # ``_log_llm_usage`` calls inside ``_call_anthropic`` /
        # ``_call_openai`` / ``_call_google`` populate ``_cumulative_usage``;
        # we diff before/after to attribute that delta to this VLM call.
        import time as _time
        _session_start = _time.time()
        _usage_before = dict(self._cumulative_usage)
        _calls_before = self._cumulative_calls
        emit_agent_lifecycle_event(
            agent=self.agent_name,
            state="started",
            model=self.model or "",
            provider=self.provider or "",
            session_kind="invoke_llm",
        )

        try:
            if self.provider == "anthropic":
                response_text, thinking_text = await self._call_anthropic(text_prompt, image_paths, operation)
            elif self.provider == "openai":
                response_text, thinking_text = await self._call_openai(text_prompt, image_paths, operation)
            elif self.provider == "google":
                response_text, thinking_text = await self._call_google(text_prompt, image_paths, operation)
            else:
                raise AgentLLMError(
                    agent_name=self.agent_name,
                    operation=operation,
                    message=f"Unsupported provider for VLM: {self.provider}",
                )
        except AgentLLMError:
            raise
        except Exception as exc:
            raise AgentLLMError(
                agent_name=self.agent_name,
                operation=operation,
                message=str(exc),
                original_exception=exc,
            ) from exc
        finally:
            _elapsed = _time.time() - _session_start
            _session_usage = _diff_usage(self._cumulative_usage, _usage_before)
            _session_calls = self._cumulative_calls - _calls_before
            emit_agent_lifecycle_event(
                agent=self.agent_name,
                state="finished",
                model=self.model or "",
                provider=self.provider or "",
                elapsed=_elapsed,
                usage=_session_usage,
                calls=_session_calls,
                session_kind="invoke_llm",
            )
            self._persist_session_stats_to_dialog(
                session_kind="invoke_llm",
                elapsed=_elapsed,
                usage=_session_usage,
                calls=_session_calls,
                turns=0,
            )

        if self.dialog_manager is not None:
            try:
                # Compose the on-disk artifact: THINKING (if any) + answer text.
                # Keeping the visible-answer JSON last preserves regex / parser
                # compatibility for callers that grep the response file for the
                # final ``{"pass": ...}`` block.
                if thinking_text:
                    body = f"# THINKING\n{thinking_text[:20000]}\n\n# RESPONSE\n{response_text}"
                    response_metadata = dict(metadata, thinking_chars=len(thinking_text))
                else:
                    body = response_text
                    response_metadata = metadata
                await asyncio.to_thread(
                    self.dialog_manager.log_response,
                    self.agent_name,
                    body,
                    response_metadata,
                )
            except Exception as exc:
                self.logger.warning(f"Failed to log VLM response ({operation}): {exc}")

        return response_text

    async def _call_anthropic(
        self, text_prompt: str, image_paths: Optional[List[str]], operation: str,
    ) -> Tuple[str, str]:
        content: list = [{"type": "text", "text": text_prompt}]
        if image_paths:
            for path in image_paths:
                b64_data, mime = await asyncio.to_thread(self._read_media_b64, path)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64_data},
                })
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        # Telemetry: ReviewAgent calls SDK clients directly (not via
        # BaseAgent.invoke_llm), so we manually feed each response through
        # ``_log_llm_usage`` here. Without this the pipeline_stats collector
        # never sees ReviewAgent's tokens.
        self._log_llm_usage(response, where=f"review_anthropic:{operation}")
        # Walk every content block. Extended-thinking responses interleave
        # ``thinking`` blocks with the visible ``text`` answer; the original
        # ``response.content[0].text`` access only worked for non-thinking
        # models and would IndexError or return the wrong block when thinking
        # was the first block.
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
        return "".join(text_parts), "".join(thinking_parts)

    async def _call_openai(
        self, text_prompt: str, image_paths: Optional[List[str]], operation: str,
    ) -> Tuple[str, str]:
        content: list = [{"type": "text", "text": text_prompt}]
        if image_paths:
            for path in image_paths:
                b64_data, mime = await asyncio.to_thread(self._read_media_b64, path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_data}"},
                })
        response = await self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        # Telemetry — see _call_anthropic above.
        self._log_llm_usage(response, where=f"review_openai:{operation}")
        msg = response.choices[0].message
        # ``reasoning_content`` is set by DeepSeek-R1, Moonshot k1.5, OpenAI
        # o1+ — None for vanilla gpt-4 / minimax non-reasoning models.
        thinking_text = getattr(msg, "reasoning_content", None) or ""
        return msg.content or "", thinking_text

    async def _call_google(
        self, text_prompt: str, image_paths: Optional[List[str]], operation: str,
    ) -> Tuple[str, str]:
        # Migrated from legacy ``google.generativeai`` to ``google.genai``.
        # Reason: the legacy SDK's ``VideoMetadata`` proto has only a
        # ``video_duration`` field — no ``fps`` — so there's no way to
        # override Gemini's default 1-fps inline-video sampling on that
        # SDK, and overriding fps is exactly what the short-clip review
        # path needs (see GEMINI_VIDEO_FPS comment).
        from google import genai
        from google.genai import types

        settings = get_settings()
        api_key = getattr(settings, "google_api_key", None)
        client = genai.Client(api_key=api_key) if api_key else genai.Client()

        parts: list = [text_prompt]
        if image_paths:
            for path in image_paths:
                b64_data, mime = await asyncio.to_thread(self._read_media_b64, path)
                import base64 as b64mod
                raw_bytes = b64mod.b64decode(b64_data)
                if mime.startswith("video/"):
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(mime_type=mime, data=raw_bytes),
                            video_metadata=types.VideoMetadata(fps=GEMINI_VIDEO_FPS),
                        )
                    )
                else:
                    parts.append(
                        types.Part(inline_data=types.Blob(mime_type=mime, data=raw_bytes))
                    )
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self.model,
            contents=parts,
        )
        # Telemetry — see _call_anthropic above. ``_extract_usage`` reads
        # ``response.usage_metadata.{prompt_token_count, candidates_token_count,
        # cached_content_token_count}`` for the google provider.
        self._log_llm_usage(response, where=f"review_google:{operation}")
        # Gemini 2.5 thinking models tag chain-of-thought parts with
        # ``part.thought = True`` when ``thinking_config`` is enabled.
        # Both ``google.generativeai`` (legacy) and ``google.genai`` expose
        # them under ``response.candidates[*].content.parts``. Most current
        # setups don't enable thought-summary streaming, so this typically
        # returns empty — which is fine; we just log no THINKING block.
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        try:
            for cand in getattr(response, "candidates", []) or []:
                cand_content = getattr(cand, "content", None)
                for part in getattr(cand_content, "parts", []) or []:
                    if getattr(part, "thought", False):
                        thinking_parts.append(getattr(part, "text", "") or "")
                    else:
                        # Visible-text part — collect as fallback if
                        # response.text aggregation misses anything.
                        ptext = getattr(part, "text", "") or ""
                        if ptext:
                            text_parts.append(ptext)
        except Exception:  # noqa: BLE001 — never let introspection break the call
            pass
        # Prefer the SDK's ``response.text`` (handles concatenation +
        # safety-block detection correctly); fall back to manual collection.
        try:
            visible_text = response.text
        except Exception:  # noqa: BLE001 — response.text raises on certain blocked outputs
            visible_text = "".join(text_parts)
        return visible_text, "".join(thinking_parts)

    # ------------------------------------------------------------------
    # Single-image description
    # ------------------------------------------------------------------

    async def _describe_single_image(
        self,
        image_path: str,
        camera_label: str,
        objectives: List[str],
        plan_summary: str,
    ) -> str:
        """Send a single camera image to VLM and return its description text."""
        prompt = review_prompts.VLM_SINGLE_IMAGE_PROMPT.substitute(
            camera_label=camera_label,
            objectives="\n".join(objectives),
            plan_summary=plan_summary,
        )

        response_text = await self._call_vlm(
            text_prompt=prompt,
            image_paths=[image_path],
            operation="review_single_image",
        )

        result = self._extract_json_from_response(response_text)
        return self._format_image_description(result)

    # ------------------------------------------------------------------
    # Step-level review / describe / decision
    # ------------------------------------------------------------------

    async def review_step(
        self,
        step_description: str,
        step_number: int,
        total_steps: int,
        completed_steps: List[str],
        plan: SimulationPlan,
    ) -> Dict[str, Any]:
        """VLM pass/fail judgment for a single scene-building step.

        Returns: {"pass": bool, "description": str, "issues": List[str]}
        """
        settings = get_settings()
        cams_dir = Path(settings.visualization_output_path) / "cam"
        images = self._collect_cam_images(cams_dir)

        if not images:
            if self._expects_camera_images(plan):
                _exp_mode = getattr(plan, "recording_mode", None) or (
                    (plan.visualization or {}).get("mode", "?")
                )
                fail_msg = (
                    f"PIPELINE FAILURE — recording_mode='{_exp_mode}' but "
                    f"ZERO mp4/png files found at {cams_dir}. The simulation "
                    "executed without producing the rendered output that "
                    "ReviewAgent reads. Most likely root causes (in order): "
                    "(1) generated code wrote the mp4 to a custom path like "
                    "'../results/vsg.mp4' instead of the canonical 'cam/vsg.mp4' "
                    "(setup_vsg_recording resolves relative paths against the "
                    "running script's directory, so 'cam/vsg.mp4' lands in "
                    f"{cams_dir} automatically); (2) generated code skipped "
                    "setup_vsg_recording / setup_preview_camera entirely; "
                    "(3) the recorder's finalize() was not called inside a "
                    "try/finally so an exception left an .inprogress file "
                    "that was filtered out. This step CANNOT pass without "
                    "the rendered output — codegen MUST fix the recording "
                    "setup before this step is re-reviewed."
                )
                self.logger.error(
                    "No camera images for step review but recording_mode='%s' "
                    "expects them — FAIL", _exp_mode,
                )
                return {
                    "pass": False,
                    "description": fail_msg,
                    "issues": [
                        f"Missing expected camera output at {cams_dir}",
                        "Use setup_vsg_recording(vis, 'cam/vsg.mp4', fps=50.0) "
                        "(literal relative path 'cam/vsg.mp4', not an absolute "
                        "or '../...' path)",
                    ],
                }
            self.logger.info("No camera images for step review (not expected in this mode) — pass")
            return {"pass": True, "description": "No camera images available (not expected).", "issues": []}

        completed_text = "\n".join(
            f"  {i}. {s}" for i, s in enumerate(completed_steps, 1)
        ) if completed_steps else "  (none — this is the first step)"

        prompt = review_prompts.STEP_REVIEW_PROMPT.substitute(
            step_number=step_number,
            total_steps=total_steps,
            step_description=step_description,
            completed_steps_text=completed_text,
        )

        response_text = await self._call_vlm(
            text_prompt=prompt,
            image_paths=images,
            operation="review_step",
        )

        result = self._extract_json_from_response(response_text)
        return {
            "pass": bool(result.get("pass", False)),
            "description": result.get("description", ""),
            "issues": result.get("issues", []),
        }

    async def describe_step_scene(
        self,
        step_description: str,
        step_number: int,
        total_steps: int,
        completed_steps: List[str],
        plan: Optional[SimulationPlan] = None,
        scene_objects_manifest: str = "(no procedural scene objects)",
        plan_assets_manifest: str = "(no plan-level external assets)",
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """VLM description-only for a single scene-building step (no pass/fail).

        ``image_paths`` is the pre-collected list of cam media (mp4 / png)
        from the workflow node. When None, fall back to internal collection
        — kept for the legacy / single-call code paths that don't preload.
        Production step_review uses the pre-collected list so the cam
        directory is scanned exactly once per step.

        Returns: {"description": str, "visible_objects": [...], "spatial_observations": [...]}
        """
        settings = get_settings()
        cams_dir = Path(settings.visualization_output_path) / "cam"
        if image_paths is None:
            images = self._collect_cam_images(cams_dir)
        else:
            images = list(image_paths)

        if not images:
            if plan and self._expects_camera_images(plan):
                _exp_mode = getattr(plan, "recording_mode", None) or (
                    (plan.visualization or {}).get("mode", "?")
                )
                # Build a description that the downstream review_step_decision
                # LLM literally cannot reasonably PASS on. The previous text
                # "No camera images produced despite sensor_camera mode" was
                # short enough that the LLM treated it as "no contradictory
                # evidence available → manifest fallback PASS" — exactly the
                # iter_002 silent-pass machine. The text below names the
                # failure as a PIPELINE FAILURE, points at the canonical
                # cam path, and is impossible to read as "no signal".
                self.logger.error(
                    "No camera images for step describe but recording_mode='%s' "
                    "expects them — emitting structural fail description",
                    _exp_mode,
                )
                return {
                    "description": (
                        f"PIPELINE FAILURE: recording_mode='{_exp_mode}' but no "
                        f"mp4/png exists at {cams_dir}. The simulation ran without "
                        "writing the rendered output ReviewAgent reads. This is a "
                        "DEFINITIVE step failure — the rendered scene cannot be "
                        "judged because nothing was rendered to the agreed location. "
                        "Codegen must use setup_vsg_recording(vis, 'cam/vsg.mp4', "
                        "fps=50.0) (literal relative path 'cam/vsg.mp4' — "
                        "setup_vsg_recording resolves it against the script "
                        "directory) or setup_preview_camera writing into 'cam/'. "
                        "Do NOT pass on the procedural-objects manifest alone — "
                        "the manifest is plan-time metadata, not evidence the "
                        "code actually built or rendered the scene correctly."
                    ),
                    "visible_objects": [],
                    "spatial_observations": [
                        f"PIPELINE FAILURE: missing rendered output at {cams_dir}",
                        "Codegen MUST write to 'cam/<name>.mp4' relative path",
                    ],
                }
            self.logger.info("No camera images for step VLM describe (not expected)")
            return {"description": "No camera images available.", "visible_objects": [], "spatial_observations": []}

        completed_text = "\n".join(
            f"  {i}. {s}" for i, s in enumerate(completed_steps, 1)
        ) if completed_steps else "  (none — this is the first step)"

        prompt = review_prompts.STEP_VLM_DESCRIBE_PROMPT.substitute(
            step_number=step_number,
            total_steps=total_steps,
            step_description=step_description,
            completed_steps_text=completed_text,
            scene_objects_manifest=scene_objects_manifest,
            plan_assets_manifest=plan_assets_manifest,
        )

        response_text = await self._call_vlm(
            text_prompt=prompt,
            image_paths=images,
            operation="describe_step_scene",
        )

        result = self._extract_json_from_response(response_text)
        # New schema (per turn 2026-04-27): describer returns ``objects`` —
        # a list of {name, present, motion_state, location} dicts. Older
        # ``spatial_observations`` is preserved as fallback for backward
        # compatibility with replay traces written before the schema swap.
        objects: List[Dict[str, Any]] = []
        for entry in result.get("objects") or []:
            if not isinstance(entry, dict):
                continue
            objects.append({
                "name": str(entry.get("name") or "").strip(),
                "present": bool(entry.get("present", False)),
                "motion_state": str(entry.get("motion_state") or "unclear"),
                "location": str(entry.get("location") or ""),
            })
        return {
            "description": result.get("description", ""),
            "visible_objects": result.get("visible_objects", []),
            "objects": objects,
            # Legacy field preserved for transcripts; empty under new schema.
            "spatial_observations": result.get("spatial_observations", []),
        }

    async def review_step_decision(
        self,
        step_description: str,
        step_number: int,
        total_steps: int,
        completed_steps: List[str],
        csv_summary: str,
        vlm_description: str,
        codegen_rebuttal: str = "",
        plan_assets_manifest: str = "(no plan-level external assets)",
        scene_objects_manifest: str = "(no procedural scene objects)",
        step_motion_expectations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Review agent final decision based on CSV results + VLM description.

        Returns: {"pass": bool, "reasoning": str, "issues": [...]}

        If ``codegen_rebuttal`` is non-empty, this is a re-evaluation triggered
        by the code agent pushing back on a previous rejection. The rebuttal is
        injected into the prompt so the review agent can reconsider.

        ``plan_assets_manifest`` lists the plan-level ``plan.assets[]`` so the
        reviewer can distinguish "forward-looking dynamics" (OK to skip per
        the CRITICAL section) from "missing static asset declared in plan"
        (NOT OK — would let a vehicle silently never spawn). Default value
        keeps the parameter optional for callers that don't have the plan
        in scope; the recommended path is to compute and pass it from the
        workflow node.
        """
        completed_text = "\n".join(
            f"  {i}. {s}" for i, s in enumerate(completed_steps, 1)
        ) if completed_steps else "  (none — this is the first step)"

        # Conflict 4 fix: surface step_motion_expectations explicitly into
        # the review prompt so rule 3a is deterministic instead of LLM-
        # inferred from narrative verbs. Empty list → static OK; non-empty
        # list → those bodies must move per the VLM table.
        _names = list(step_motion_expectations or [])
        if _names:
            motion_expectations_block = (
                "  These bodies MUST be reported with motion_state ∈ "
                "{moving, falling, oscillating} in the VLM observation "
                "table; any of them reporting motion_state=static is a "
                "FAIL:\n"
                + "\n".join(f"    - {n}" for n in _names)
            )
        else:
            motion_expectations_block = (
                "  (empty — this is a setup / static / braked / build-only "
                "step; every body is allowed to be motion_state=static)"
            )

        rebuttal_block = ""
        if codegen_rebuttal.strip():
            rebuttal_block = (
                "\n## Code-agent rebuttal (from previous review cycle)\n"
                "The code agent disagrees with the previous review decision and "
                "claims the code is ALREADY correct. Their reasoning:\n\n"
                f"{codegen_rebuttal.strip()}\n\n"
                "You must reconsider your decision in light of this rebuttal. If "
                "the rebuttal correctly identifies that the previous review was "
                "based on a false positive (e.g. you misread which axis the "
                "asset's canonical front points at, you over-interpreted the "
                "VLM description, or the CSV already matches the plan), accept "
                "the rebuttal by returning pass=true. If the rebuttal is still "
                "wrong on the merits, return pass=false and explain WHY the "
                "rebuttal is incorrect so the code agent has something concrete "
                "to fix.\n"
            )

        # Judge runs text-only. The mp4 has already been read by
        # ``describe_step_scene`` (the only VLM pass that gets pixels);
        # the judge weighs that narration + csv_summary + manifests. If
        # the describer hit a pipeline failure (no cam media), it
        # already injected a PIPELINE FAILURE marker into
        # ``vlm_description`` so the text-only judgment still produces
        # the correct fail.
        prompt = review_prompts.STEP_REVIEW_DECISION_PROMPT.substitute(
            step_number=step_number,
            total_steps=total_steps,
            step_description=step_description,
            step_motion_expectations_block=motion_expectations_block,
            completed_steps_text=completed_text,
            csv_summary=csv_summary,
            vlm_description=vlm_description,
            rebuttal_block=rebuttal_block,
            plan_assets_manifest=plan_assets_manifest,
            scene_objects_manifest=scene_objects_manifest,
        )

        response_text = await self._call_vlm(
            text_prompt=prompt,
            image_paths=None,
            operation="review_step_decision",
        )

        result = self._extract_json_from_response(response_text)
        return {
            "pass": bool(result.get("pass", False)),
            "reasoning": result.get("reasoning", ""),
            "issues": result.get("issues", []),
        }

    # ------------------------------------------------------------------
    # Unchanged helpers
    # ------------------------------------------------------------------

    def _expects_camera_images(self, plan: SimulationPlan) -> bool:
        """Return True when the plan REQUIRES rendered output for review.

        Two independent signals — either one means the simulation should
        produce a ``cam/<name>.mp4`` (or PNG sequence) that ReviewAgent will
        read at ``<visualization_output_path>/cam/``:

          1. ``plan.visualization.mode`` ∈ {sensor_camera,
             vsg_with_sensor_camera} — sensor-camera path
             (`setup_preview_camera` produces the mp4).
          2. ``plan.recording_mode == "vsg_only"`` — VSG-only path
             (`setup_vsg_recording` produces the mp4). This signal was
             missing from the original implementation; FSI / SPH plans
             default to vsg_only mode and were silently treated as
             "headless, no images expected" → review fell back to a
             manifest-only PASS even when the actual mp4 was missing or
             the codegen wrote it to the wrong directory.

        Returning True here flips the no-images branch in
        ``review_step`` / ``describe_step_scene`` from
        "silently pass" to "explicit fail with actionable feedback for
        codegen".
        """
        vis_mode = (plan.visualization or {}).get("mode", "headless")
        recording_mode = getattr(plan, "recording_mode", None) or "sensor_cams"
        return (
            vis_mode in ("sensor_camera", "vsg_with_sensor_camera")
            or recording_mode == "vsg_only"
        )

    def _collect_cam_images(self, cams_dir: Path) -> List[str]:
        """Collect camera media from cam/.

        For the Google provider, mp4 files are returned directly (Gemini accepts
        native video). For other providers, only PNG/JPG images are returned.

        **Inprogress filter**: ``setup_preview_camera`` writes to
        ``<name>.inprogress.mp4`` while the recorder is live, then
        atomically renames to ``<name>.mp4`` on ``recorder.release()``.
        When execution is interrupted (timeout / SIGKILL), the inprogress
        sibling survives. Its ``Path.suffix`` is ``.mp4`` so a naive
        suffix filter picks it up — but the file has no ``moov`` atom
        and VLM calls fail opening it. We exclude anything with
        ``.inprogress`` anywhere in the filename stem.
        """
        if not cams_dir.is_dir():
            self.logger.warning(f"Camera directory not found: {cams_dir}")
            return []

        is_google = self.provider == "google"
        media: List[str] = []
        skipped_inprogress: List[str] = []
        for f in sorted(cams_dir.iterdir()):
            suffix = f.suffix.lower()
            # Reject any file whose stem flags it as mid-recording.
            # ``Path.stem`` drops only the final extension, so
            # ``cam_0.inprogress.mp4`` → stem ``cam_0.inprogress``.
            if ".inprogress" in f.stem.lower():
                skipped_inprogress.append(f.name)
                continue
            if suffix in IMAGE_EXTENSIONS:
                media.append(str(f))
            elif is_google and suffix in VIDEO_EXTENSIONS:
                media.append(str(f))

        if skipped_inprogress:
            self.logger.warning(
                "Skipped %d unfinalized recording(s) in %s: %s — this "
                "typically means the simulation process was killed "
                "mid-run before recorder.release() could atomic-rename "
                ".inprogress.mp4 to .mp4.",
                len(skipped_inprogress), cams_dir,
                ", ".join(skipped_inprogress),
            )

        # Issue 3: Gemini handles mp4 natively (internal 5 fps sampling).
        # When we have an mp4 AND a PNG sequence on the Google path, they
        # encode the same content — sending both doubles the input tokens
        # for no review benefit. Drop the PNGs unless the operator has
        # explicitly opted out via vlm_prefer_native_video=False.
        if is_google:
            try:
                settings = get_settings()
                prefer_native = bool(getattr(settings, "vlm_prefer_native_video", True))
            except Exception:
                prefer_native = True
            mp4s = [p for p in media if Path(p).suffix.lower() in VIDEO_EXTENSIONS]
            pngs = [p for p in media if Path(p).suffix.lower() in IMAGE_EXTENSIONS]
            if prefer_native and mp4s and pngs:
                self.logger.info(
                    "native video, dropping %d redundant png(s) (kept %d mp4)",
                    len(pngs), len(mp4s),
                )
                media = mp4s

        # Optional opt-in PNG-sequence sampling (default off). Kept disabled
        # until somebody measures whether sampled review catches the same
        # failures as full review.
        try:
            settings = get_settings()
            if bool(getattr(settings, "vlm_frame_sampling_enabled", False)):
                max_frames = int(getattr(settings, "vlm_max_frames", 12))
                stride_cfg = getattr(settings, "vlm_frame_stride", "auto")
                media = self._sample_images(media, max_frames=max_frames, stride=stride_cfg)
        except Exception as exc:
            self.logger.debug(f"Frame sampling skipped: {exc}")

        self.logger.info(f"Found {len(media)} camera media files in {cams_dir}")
        return media

    @staticmethod
    def _sample_images(
        media: List[str],
        *,
        max_frames: int,
        stride: Any = "auto",
    ) -> List[str]:
        """Stride-sample PNG/JPG frames, preserving first + last and any mp4s.

        mp4 entries always pass through (Gemini samples them internally).
        For image entries: stride is auto-derived to land near ``max_frames``
        when stride=="auto"; an integer stride is honored as-is. Result is
        capped at ``max_frames`` images, always keeping the first and last
        frame so endpoint events aren't lost.
        """
        if not media:
            return media
        videos = [p for p in media if Path(p).suffix.lower() in VIDEO_EXTENSIONS]
        images = [p for p in media if Path(p).suffix.lower() in IMAGE_EXTENSIONS]
        if not images or max_frames <= 0:
            return videos + images
        if isinstance(stride, str) and stride.strip().lower() == "auto":
            step = max(1, math.ceil(len(images) / max_frames))
        else:
            try:
                step = max(1, int(stride))
            except (TypeError, ValueError):
                step = 1
        sampled = images[::step]
        if images[-1] not in sampled:
            sampled = sampled + [images[-1]]
        if len(sampled) > max_frames:
            head = sampled[: max_frames - 1]
            sampled = head + [sampled[-1]]
        return videos + sampled

    @staticmethod
    def detect_incomplete_recording(cams_dir: Path) -> Dict[str, Any]:
        """Characterize the ``cam/`` directory so upstream workflow nodes
        can tell "observer failed" from "execution produced no video".

        Returns a dict with:
          * ``has_final``: at least one finalized ``.mp4`` / ``.png`` present
          * ``inprogress_names``: list of ``*.inprogress.*`` filenames left behind
          * ``incomplete``: True iff inprogress files exist AND no final
            output — this is the signal that the simulation was killed
            mid-run and the correct recovery is to re-execute, NOT to
            blame codegen.
        """
        if not cams_dir.is_dir():
            return {"has_final": False, "inprogress_names": [], "incomplete": False}
        inprogress: List[str] = []
        has_final = False
        for f in cams_dir.iterdir():
            if ".inprogress" in f.stem.lower():
                inprogress.append(f.name)
                continue
            if f.suffix.lower() in IMAGE_EXTENSIONS or f.suffix.lower() in VIDEO_EXTENSIONS:
                has_final = True
        return {
            "has_final": has_final,
            "inprogress_names": sorted(inprogress),
            "incomplete": bool(inprogress) and not has_final,
        }

    @staticmethod
    def _is_video(path: str) -> bool:
        return Path(path).suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def _read_media_b64(path: str) -> tuple:
        """Read a media file and return (base64_data, mime_type)."""
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        suffix = Path(path).suffix.lower()
        if suffix in _MIME_MAP_VIDEO:
            mime = _MIME_MAP_VIDEO[suffix]
        else:
            mime = _MIME_MAP_IMAGE.get(suffix, "image/png")
        return data, mime

    def _media_block(self, path: str) -> dict:
        """Build a provider-appropriate content block for an image or video file."""
        b64_data, mime = self._read_media_b64(path)

        if self.provider == "anthropic":
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64_data},
            }
        elif self.provider == "google":
            if mime.startswith("video/"):
                return {"type": "media", "mime_type": mime, "data": b64_data}
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64_data},
            }
        else:
            # OpenAI-compatible format (default)
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64_data}"},
            }

    @staticmethod
    def _format_image_description(result: dict) -> str:
        """Format a single image's VLM JSON result into readable text."""
        lines = []

        desc = result.get("description", "")
        if desc:
            lines.append(desc)

        camera_view = result.get("camera_view")
        if camera_view:
            lines.append(f"Camera view: {camera_view}")

        visible_objects = result.get("visible_objects") or []
        if visible_objects:
            lines.append("Visible objects: " + ", ".join(str(x) for x in visible_objects))

        observations = result.get("observations") or []
        if observations:
            lines.append("Observations: " + "; ".join(str(x) for x in observations))

        return "\n".join(lines) if lines else "No description produced."

    def _build_plan_summary(self, plan: SimulationPlan) -> str:
        """Create concise plan summary for review-time context injection."""
        lines = [f"Plan type: {plan.plan_type}"]

        if plan.objectives:
            lines.append("Objectives:")
            for idx, obj in enumerate(plan.objectives[:8], start=1):
                lines.append(f"{idx}. {obj}")

        if plan.implementation_steps:
            lines.append("Key implementation steps:")
            for idx, step in enumerate(plan.implementation_steps[:8], start=1):
                lines.append(f"{idx}. {step}")

        if plan.simulation_parameters:
            lines.append("Key simulation parameters:")
            for key in sorted(plan.simulation_parameters.keys())[:12]:
                lines.append(f"- {key}: {plan.simulation_parameters[key]}")

        return "\n".join(lines)

    @staticmethod
    def _read_image_b64(path: str) -> str:
        """Read image file and return base64 string."""
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode()

    def _extract_json_from_response(self, response_content: str) -> dict:
        """Extract JSON from VLM response, handling markdown code blocks."""
        import json

        content = response_content.strip()

        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            self.logger.warning(f"Failed to parse VLM response as JSON: {content[:200]}")
            raise AgentLLMError(
                agent_name=self.agent_name,
                operation="review_parse_json",
                message=f"Failed to parse VLM JSON response: {exc}",
                original_exception=exc,
            ) from exc
