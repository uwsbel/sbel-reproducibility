"""PlanningAgent — thin shell over the 6-phase pipeline. See plan_agent.md."""

import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from chrono_agent.agents.base import BaseAgent
from chrono_agent.models.clarification import StructuredClarification
from chrono_agent.models.plan import SimulationPlan


# Phase 4 batch callback signature: List[StructuredClarification] → Dict[target_field, answer].
BatchCallback = Callable[
    [List[StructuredClarification]], Awaitable[Dict[str, Any]]
]


class PlanningAgent(BaseAgent):
    """Plan-generation agent. Wraps the 6-phase pipeline."""

    # Wrapper-backed wheeled vehicles — no-arg Python factories in pychrono.vehicle.
    _WRAPPER_VEHICLES: tuple = (
        ("hmmwv", "veh.HMMWV_Full()", ("hmmwv_full", "humvee", "m998")),
        ("hmmwv_reduced", "veh.HMMWV_Reduced(sys)", ("reduced_hmmwv",)),
        ("citybus", "veh.CityBus()", ("city_bus",)),
        ("feda", "veh.FEDA()", ()),
    )

    # Chrono built-in scene asset categories scanned from GetChronoDataPath().
    _CHRONO_SCENE_CATEGORIES = (
        "models",
        "sensor/offroad",
        "sensor/geometries",
        "sensor/cones",
    )

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        clarification_callback: Optional[BatchCallback] = None,
    ):
        """``clarification_callback`` is the Phase 4 batch UI callback."""
        super().__init__(
            agent_name="PlanningAgent",
            agent_number=1,
            llm_provider=llm_provider,
            model=model,
            temperature=temperature,
        )
        self._clarification_callback = clarification_callback

    # ---- Public API --------------------------------------------------------

    async def execute(
        self,
        user_prompt: str,
        plan_mode: str = "auto",
        images: Optional[List[Union[str, Path]]] = None,
    ) -> SimulationPlan:
        """Run the 6-phase pipeline and return a validated plan.

        ``plan_mode`` is accepted for back-compat but ignored — the pipeline
        derives plan_type from the user prompt and asset classification.
        """
        from chrono_agent.agents.planning_pipeline import run_pipeline

        self.logger.info("Planning prompt: %s...", (user_prompt or "")[:100])
        if images:
            self.logger.info("Planning with %d image(s)", len(images))
        return await run_pipeline(
            agent=self,
            user_prompt=user_prompt,
            images=images,
            clarification_batch_callback=self._clarification_callback,
        )

    async def modify_plan(
        self,
        original_plan: SimulationPlan,
        modification_request: str,
        images: Optional[List[Union[str, Path]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,  # noqa: ARG002
    ) -> SimulationPlan:
        """Phase 6: apply user modification text to an existing plan."""
        from chrono_agent.agents.planning_pipeline import run_modify

        self.logger.info("Modifying plan: %s...", (modification_request or "")[:80])
        return await run_modify(
            agent=self,
            current_plan=original_plan,
            modification_text=modification_request,
            images=images,
            clarification_batch_callback=self._clarification_callback,
        )

    async def refine_plan(
        self,
        original_plan: SimulationPlan,
        user_response: str,
        images: Optional[List[Union[str, Path]]] = None,
    ) -> SimulationPlan:
        """Refinement = modification with the user's clarification answer text."""
        return await self.modify_plan(
            original_plan=original_plan,
            modification_request=user_response,
            images=images,
        )

    # ---- Helpers -----------------------------------------------------------

    def _image_to_content_block(self, image: Union[str, Path]) -> dict:
        """Convert a local path / http(s) URL to an OpenAI-style image block."""
        import base64

        s = str(image)
        if s.startswith(("http://", "https://", "data:")):
            return {"type": "image_url", "image_url": {"url": s}}
        path = Path(s)
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(path.suffix.lower(), "image/jpeg")
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}

    @staticmethod
    def _looks_like_scene(user_prompt: str) -> bool:
        """Heuristic for whether a user prompt describes a bounded scene
        (so asking for ``scene_size`` makes sense) vs a pure mechanism
        (pendulum, gear train, link/joint demo) where a scene size would
        be off-topic.

        Returns True if the prompt mentions any spatial-context keyword.
        Used by ``_ask_critical_clarifications`` in main.py to decide
        whether to prompt the user for ``scene_size``.
        """
        keywords = (
            "room", "indoor", "outdoor", "terrain", "scene",
            "floor", "ground", "environment",
            "courtyard", "arena", "field", "yard", "plaza",
            "garage", "hall", "house", "cottage",
            "wall", "tank", "pool", "basin", "container",
        )
        s = (user_prompt or "").lower()
        return any(k in s for k in keywords)

    @staticmethod
    def _get_chrono_data_path() -> Optional[Path]:
        """Chrono built-in data root, or None if not installed."""
        import sys as _sys

        try:
            import pychrono.core as chrono
            p = Path(chrono.GetChronoDataPath())
            if p.is_dir():
                return p
        except Exception:
            pass
        env = os.environ.get("CHRONO_DATA_DIR")
        if env:
            p = Path(env)
            if p.is_dir():
                return p
        p = Path(_sys.prefix) / "share" / "chrono" / "data"
        return p if p.is_dir() else None

    @classmethod
    def _scan_asset_catalog(cls) -> Dict[str, Dict[str, Any]]:
        """Walk data/scene/, data/robot/, data/vehicle/, and Chrono built-ins.

        Project assets use repo-relative filenames (``data/scene/foo/foo.obj``).
        Chrono built-ins use chrono-data-relative filenames
        (``sensor/offroad/tree1.obj``), matching ``chrono.GetChronoDataFile()``.
        Vehicle JSONs use GetVehicleDataFile-relative form (``Polaris/Polaris.json``).
        """
        catalog: Dict[str, Dict[str, Any]] = {}

        for root_label, root in (("scene", Path("data/scene")), ("robot", Path("data/robot"))):
            if not root.is_dir():
                continue
            for asset_dir in sorted(root.iterdir()):
                if not asset_dir.is_dir():
                    continue
                urdf = sorted(asset_dir.rglob("*.urdf"))
                obj = sorted(asset_dir.glob("*.obj"))
                if urdf:
                    primary, asset_type = urdf[0], "urdf"
                elif obj:
                    primary, asset_type = obj[0], "mesh"
                else:
                    continue
                catalog[asset_dir.name] = {
                    "name": asset_dir.name,
                    "filename": str(primary).replace("\\", "/"),
                    "type": asset_type,
                    "category": root_label,
                    "source": "project",
                }

        # Wrapper-backed wheeled vehicles (factory expressions, no filename).
        for slug, factory, aliases in cls._WRAPPER_VEHICLES:
            catalog[slug] = {
                "name": slug,
                "filename": None,
                "factory": factory,
                "aliases": list(aliases),
                "type": "wrapper_vehicle",
                "category": "vehicle",
                "source": "chrono_wrapper",
                "is_dynamic": True,
            }

        # JSON-driven vehicles in data/vehicle/. Skip dirs that overlap a wrapper.
        wrapper_tokens: set = set()
        for slug, _factory, aliases in cls._WRAPPER_VEHICLES:
            wrapper_tokens.update(cls._asset_tokens(slug))
            for alias in aliases:
                wrapper_tokens.update(cls._asset_tokens(alias))

        vehicle_root = Path("data/vehicle")
        if vehicle_root.is_dir():
            for veh_dir in sorted(vehicle_root.iterdir()):
                if not veh_dir.is_dir():
                    continue
                dir_tokens = cls._asset_tokens(veh_dir.name)
                if dir_tokens and dir_tokens.issubset(wrapper_tokens):
                    continue
                primary_json: Optional[Path] = None
                dir_token = veh_dir.name.lower()
                for json_file in sorted(veh_dir.glob("*.json")):
                    if json_file.stem.lower() == dir_token:
                        primary_json = json_file
                        break
                if primary_json is None:
                    continue
                catalog[veh_dir.name] = {
                    "name": veh_dir.name,
                    "filename": f"{veh_dir.name}/{primary_json.name}",
                    "type": "vehicle_json",
                    "category": "vehicle",
                    "source": "project",
                    "is_dynamic": True,
                }

        chrono_root = cls._get_chrono_data_path()
        if chrono_root is not None:
            for cat in cls._CHRONO_SCENE_CATEGORIES:
                cat_dir = chrono_root / cat
                if not cat_dir.is_dir():
                    continue
                for obj_file in sorted(cat_dir.glob("*.obj")):
                    if not obj_file.is_file():
                        continue
                    rel = str(obj_file.relative_to(chrono_root)).replace("\\", "/")
                    catalog[f"chrono:{obj_file.stem}"] = {
                        "name": obj_file.stem,
                        "filename": rel,
                        "type": "mesh",
                        "category": "chrono_builtin",
                        "source": "chrono",
                    }

        return catalog

    @staticmethod
    def _asset_tokens(s: str) -> set:
        """Tokenize an asset name on letter/digit boundaries, lowercased."""
        return set(re.findall(r"[a-z]+|[0-9]+", (s or "").lower()))
