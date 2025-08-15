from collections import deque

class TrieNode:
    def __init__(self):
        self.children = {}
        self.suggestions = []
        self.is_end = False

class OptimizedTrie:
    def __init__(self):
        self.root = TrieNode()
        self.suggestion_scores = {}  # For ranking suggestions
    
    def insert(self, suggestion, score=1.0):
        """Insert suggestion with optional scoring for ranking."""
        node = self.root
        suggestion_lower = suggestion.lower()
        
        # Store score for ranking
        self.suggestion_scores[suggestion] = score
        
        for char in suggestion_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            
            # Keep top suggestions at each node, sorted by score
            if suggestion not in [s[0] for s in node.suggestions]:
                node.suggestions.append((suggestion, score))
                node.suggestions.sort(key=lambda x: x[1], reverse=True)
                node.suggestions = node.suggestions[:15]  # Keep top 15 per node
        
        node.is_end = True
    
    def search_prefix(self, prefix, max_results=20):
        """Search with prefix and return ranked results."""
        if not prefix:
            return []
            
        node = self.root
        prefix_lower = prefix.lower()
        
        # Navigate to prefix node
        for char in prefix_lower:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect suggestions using BFS for better distribution
        results = []
        queue = deque([(node, "")])
        visited_suggestions = set()
        
        while queue and len(results) < max_results:
            current_node, path = queue.popleft()
            
            # Add suggestions from current node
            for suggestion, score in current_node.suggestions:
                if (suggestion not in visited_suggestions and 
                    len(results) < max_results and
                    suggestion.lower().startswith(prefix_lower)):
                    results.append(suggestion)
                    visited_suggestions.add(suggestion)
            
            # Add child nodes to queue
            for char, child_node in current_node.children.items():
                if len(results) < max_results:
                    queue.append((child_node, path + char))
        
        return results[:max_results]