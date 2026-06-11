class FusedFeatureExchangeManager:
    def __init__(self):
        self.buffers = {}  # agent_id → list of incoming feature tensors

    def register_agent(self, agent_id: int):
        if agent_id in self.buffers:
            raise ValueError(f"agent {agent_id} already registered")
        self.buffers[agent_id] = []

    def push(self, sender_id: int, feat):
        # send feat to every other agent’s inbox
        for aid, buf in self.buffers.items():
            if aid != sender_id:
                buf.append(feat)

    def retrieve(self, receiver_id: int):
        feats = list(self.buffers.get(receiver_id, []))
        self.buffers[receiver_id] = []
        return feats

    def reset(self):
        for agent_id in self.buffers:
            self.buffers[agent_id] = []
