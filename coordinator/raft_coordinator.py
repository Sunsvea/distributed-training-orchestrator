import asyncio
import time
import random
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from communication.cluster_pb2 import *

@dataclass
class RaftCoordinator:
    node_id: str
    address: str
    port: int
    cluster_nodes: List[str]
    
    # Raft state
    state: str = "follower"  # follower, candidate, leader
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[LogEntry] = field(default_factory=list)
    commit_index: int = 0
    last_applied: int = 0
    
    # Leader state
    leader_id: Optional[str] = None
    next_index: Dict[str, int] = field(default_factory=dict)
    match_index: Dict[str, int] = field(default_factory=dict)
    
    # Election state
    votes_received: int = 0
    election_timeout: float = 0.0
    heartbeat_interval: float = 0.1
    
    def __post_init__(self):
        self.reset_election_timeout()
    
    def reset_election_timeout(self):
        self.election_timeout = time.time() + random.uniform(0.5, 1.0)
    
    def start_election(self):
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = 1
        self.reset_election_timeout()
    
    def handle_vote_request(self, request: VoteRequest) -> VoteResponse:
        if request.term < self.current_term:
            return VoteResponse(term=self.current_term, vote_granted=False)
        
        if request.term > self.current_term:
            self.current_term = request.term
            self.voted_for = None
            self.state = "follower"
        elif request.term == self.current_term:
            # If we already voted for someone else in this term, reject
            if self.voted_for is not None and self.voted_for != request.candidate_id:
                return VoteResponse(term=self.current_term, vote_granted=False)
        
        vote_granted = False
        if (self.voted_for is None or self.voted_for == request.candidate_id) and \
           self._is_log_up_to_date(request.last_log_index, request.last_log_term):
            self.voted_for = request.candidate_id
            vote_granted = True
            self.reset_election_timeout()
        
        return VoteResponse(term=self.current_term, vote_granted=vote_granted)
    
    def handle_append_entries(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        if request.term < self.current_term:
            return AppendEntriesResponse(term=self.current_term, success=False)
        
        if request.term > self.current_term:
            self.current_term = request.term
            self.voted_for = None
        
        self.state = "follower"
        self.leader_id = request.leader_id
        self.reset_election_timeout()
        
        # Check if log matches
        if request.prev_log_index > 0:
            if len(self.log) < request.prev_log_index or \
               self.log[request.prev_log_index - 1].term != request.prev_log_term:
                return AppendEntriesResponse(term=self.current_term, success=False)
        
        # Append new entries
        if request.entries:
            # Remove conflicting entries
            if len(self.log) > request.prev_log_index:
                self.log = self.log[:request.prev_log_index]
            
            # Append new entries
            self.log.extend(request.entries)
        
        # Update commit index
        if request.leader_commit > self.commit_index:
            self.commit_index = min(request.leader_commit, len(self.log))
        
        return AppendEntriesResponse(term=self.current_term, success=True)
    
    def become_leader(self):
        self.state = "leader"
        self.leader_id = self.node_id
        
        # Initialize leader state
        self.next_index = {}
        self.match_index = {}
        
        for node in self.cluster_nodes:
            if node != f"{self.address}:{self.port}":
                self.next_index[node] = len(self.log) + 1
                self.match_index[node] = 0
    
    def _is_log_up_to_date(self, last_log_index: int, last_log_term: int) -> bool:
        if not self.log:
            return True
        
        last_entry = self.log[-1]
        if last_log_term > last_entry.term:
            return True
        elif last_log_term == last_entry.term:
            return last_log_index >= len(self.log)
        else:
            return False
    
    def append_log_entry(self, command: str, data: bytes):
        entry = LogEntry(
            term=self.current_term,
            command=command,
            data=data
        )
        self.log.append(entry)
        return len(self.log) - 1
    
    def is_election_timeout(self) -> bool:
        return time.time() > self.election_timeout
    
    def get_state(self) -> Dict:
        return {
            "node_id": self.node_id,
            "state": self.state,
            "current_term": self.current_term,
            "leader_id": self.leader_id,
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "votes_received": self.votes_received if self.state == "candidate" else 0
        }