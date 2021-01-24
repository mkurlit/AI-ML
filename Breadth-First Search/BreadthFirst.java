import java.util.ArrayList;

public class BreadthFirst {
    // Frontier 
	public ArrayList<Node> frontier = new ArrayList<Node>();
	
	// Explored -> list of state ids
	public ArrayList<Integer> explored = new ArrayList<Integer>();
	
	// Store all potential states
	public ArrayList<Node> nodeStates = new ArrayList<Node>();
	
	//Constructor - initialise -> determine all states
	public BreadthFirst() {
		
		// State 1
		this.nodeStates.add(new Node(true, true, 'L', 1,2,3,1));
		// State 2
		this.nodeStates.add(new Node(true, true, 'R', 1,2,6,2));
		// State 3
		this.nodeStates.add(new Node(false, true, 'L', 3,4,3,3));
		// State 4
		this.nodeStates.add(new Node(false, true, 'R', 3,4,8,4));
		// State 5
		this.nodeStates.add(new Node(true, false, 'L', 5,6,7,5));
		// State 6
		this.nodeStates.add(new Node(true, false, 'R', 5,6,6,6));
		// State 7
		this.nodeStates.add(new Node(false, false, 'L', 7,8,7,7));
		// State 8
		this.nodeStates.add(new Node(false, false, 'R', 7,8,8,8));
		
		//Point to the initial state
		this.frontier.add(this.nodeStates.get(0));
		
	}
	
	// Expand specified node
	public int expand(Node expNode) {
		
		//List of states to be expanded
		Integer[] nextStates = {expNode.actionLeft, expNode.actionRight, expNode.actionSuck};
		
		// Loop through the list of states from the node
		for (Integer n : nextStates){
			
			// verify if the state has already been explored or is already in the frontier
			if (! this.explored.contains(n) && !this.frontier.contains(nodeStates.get(n-1))) {
				// verify if the state is not the goal state
				if (goalState(nodeStates.get(n-1)) == false) {
					this.frontier.add(nodeStates.get(n-1));
				}
				// if it is the goal state, then add and end the expansion
				else {
					this.explored.add(expNode.id);
					this.frontier.remove(expNode);
					this.explored.add(n);
					return n;
				}	
            }
            else{
                continue;
            }
		}
		// if no goal state has been observed, add the current node to explored and remove from the frontier
		this.explored.add(expNode.id);
		this.frontier.remove(expNode);
		return -1;
	}
	
	//traverse through the graph of potential states
	public ArrayList<Integer> traverse() {
		
		//run until elements are in the frontier
		while (this.frontier.size() >= 1) {
			
			//try to expand and test goal state
			int expandResult = this.expand(this.frontier.get(0));
			if (expandResult != -1) {
				return this.explored;
			}
			else {
				continue;
			}
		
		}
		// return list of explored states and invalid # at the end to indicate failure
		this.explored.add(-1);
		return this.explored;
	}
	
	//test for the goal state
	public boolean goalState(Node testNode) {
		
		//check if there is no dirt in either left / right plane
		if (testNode.stateLeft == false && testNode.stateRight == false) {
			return true;
		}
		else {
			return false;
		}
	}
	
	//create a node object
	class Node {
		
		public boolean stateLeft;
		public boolean stateRight;
		public char vacuum;
		public int actionLeft;
		public int actionRight;
		public int actionSuck;
		public int id;
		
		//constructor of the node object
		public Node(boolean stateLeft, boolean stateRight
				,char vacuum, int actionLeft, int actionRight, int actionSuck, int id) {
			
			this.stateLeft = stateLeft;
			this.stateRight = stateRight;
			this.vacuum = vacuum;
			this.actionLeft = actionLeft;
			this.actionRight = actionRight;
			this.actionSuck = actionSuck;
			this.id = id;
		}
    }
    
    public static void main(String[] args) {
		
		BreadthFirst test = new BreadthFirst();
        ArrayList<Integer> path =  new ArrayList<Integer>();
        path = test.traverse();
		if (path.get(path.size()-1) == -1){
			System.out.println("Problem unresolved with traversed states: " + path.toString());
		}
		else{
			System.out.println("Goal state encountered with the following path: " + path.toString());
		}
		
	}
}
