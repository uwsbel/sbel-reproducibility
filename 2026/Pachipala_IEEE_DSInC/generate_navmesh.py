import habitat_sim

scene_path = "data/scene_datasets/hm3d_v0.2/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb"
output_path = "data/scene_datasets/hm3d_v0.2/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.navmesh"

# --- Navmesh parameters ---
settings = habitat_sim.NavMeshSettings()
settings.set_defaults()
settings.agent_height = 1.5
settings.agent_radius = 0.1
settings.cell_height = 0.05
settings.cell_size = 0.05

# --- Simulator configuration ---
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.enable_physics = False
sim_cfg.create_renderer = False  # ✅ fully headless
sim_cfg.scene_light_setup = "no_lights"

# Minimal agent config (needed to init)
agent_cfg = habitat_sim.agent.AgentConfiguration()

print("Initializing simulator (headless) for:", scene_path)
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

print("Building navmesh (CPU-only)...")
success = sim.recompute_navmesh(sim.pathfinder, settings)
if success:
    sim.pathfinder.save_nav_mesh(output_path)
    print("✅ Navmesh successfully saved at:", output_path)
else:
    print("⚠️ Navmesh build failed — check scene geometry or permissions")

sim.close()
