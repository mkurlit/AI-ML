import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Hashtable;

class IDSearch {

    public Hashtable<Integer, ArrayList<SearchNode>> frontier;
    public Hashtable<Integer, ArrayList<SearchNode>> scanned;
    public int[][] explored;
    public int[][] world;
    public int[][] finalRoute;
    public List<SearchNode> currentBranch;
    public SearchNode startNode;
    public SearchNode goalNode;
    public int nodeCount;
    public int start;
    public int goal;
    public int traffic;

    public IDSearch(String worldFile, int start, int goal, int traffic){
        //Initialise frontier as a hash table
        this.frontier = new Hashtable<Integer, ArrayList<SearchNode>>();
        //Initialise scanned set
        this.scanned = new Hashtable<Integer, ArrayList<SearchNode>>();
        //Read in the world for the agent
        this.world = readWorld(worldFile);
        //Create a 2d array to map explored locations
        this.explored = new int[this.world.length][this.world[0].length];
        //Create a 2d array to show final path
        this.finalRoute = readWorld(worldFile);

        //read in world markers
        this.start = start;
        this.goal = goal;
        this.traffic = traffic;
        
        //Start and goal nodes
        Hashtable<String, SearchNode> hh = getStartGoalNodes(this.world, this.start, this.goal);
        this.startNode = hh.get("Start");
        this.goalNode = hh.get("Goal");

        //Add Start goal to the frontier
        this.frontier.put(0, new ArrayList<SearchNode>(Arrays.asList(this.startNode)));

        //Trace the path to the goal
        this.currentBranch = new ArrayList<SearchNode>();

        //Set node counter 
        this.nodeCount = 0;
    }

    // Absorb csv grid from file
    public int[][] readWorld(String fileName){

        int[][] matrix = null;
        
        try {
            // Read in file
            BufferedReader buff = new BufferedReader(new FileReader(fileName));
            // List of rows
            ArrayList<int[]> store = new ArrayList<int[]>();
            
            String lineIn;
            while ((lineIn = buff.readLine()) != null) {
                String[] row_str = lineIn.trim().split(", ");
                int[] row_vals = new int[row_str.length];
                for (int i = 0; i< row_vals.length; i++){
                    row_vals[i] = Integer.parseInt(row_str[i]);
                }
                store.add(row_vals);
            }
            //Close buffer
            buff.close();
            // Convert to 2d array
            matrix = new int[store.size()][];
            for (int i = 0; i< store.size(); i++){
                matrix[i] = store.get(i);
            }
        } catch (FileNotFoundException e1){
            e1.printStackTrace();
        } catch (IOException e2) {
            e2.printStackTrace();
        }
        return matrix;
    }

    // Identify Start and Goal nodes
    public Hashtable<String, SearchNode> getStartGoalNodes(int[][] world, int start_code, int goal_code){

        //Initialise output hashtable
        Hashtable<String, SearchNode> keyNodes = new Hashtable<String, SearchNode>();

        int row = 0;
        for (int[] r : world){
            int col = 0;
            for (int val : r){
                if (val == start_code){
                    keyNodes.put("Start", new SearchNode(row, col, 0, val));
                }
                else if (val == goal_code){
                    keyNodes.put("Goal", new SearchNode(row, col, 0, val));
                }
                col++;
            }
            row++;
        }
        return keyNodes;
    }

    //Identify direction that is being taken
    public String checkDirection(SearchNode one, SearchNode two){
        int h = Integer.signum(one.y - two.y);
        int v = Integer.signum(one.x - two.x);

        if (h == 0 && v == 1){
            return "S";
        }
        else if (h==0 && v == -1){
            return "N";
        }
        else if (h==1 && v == 0){
            return "E";
        }
        else{
            return "W";
        }
    }

    //check if node exists in a list
    public boolean checkContains(List<SearchNode> l, SearchNode node){

        for (SearchNode n : l){
            if (n.checkEqualNodes(node)){
                return true;
            }
        }
        return false;
    }
    //get manhattan distance
    public int manhattanDistance(SearchNode node){
        return Math.abs(this.startNode.x - node.x) + Math.abs(this.startNode.y - node.y);
    }

    //check if the node has been already explored at current depth
    public boolean checkExplored(SearchNode node, int level){
        int manDist = this.manhattanDistance(node);
        //if minimum distance from the start node then return ok
        if (manDist == level){
            return false;
        }
        else{
            for (int k = level-1; k > manDist; k--){
                if (this.checkContains(this.scanned.get(k), node)){
                    return true;
                }
            }
            return false;
        }
    }

    //Draw the successful traversal on top of the world map
    public void drawPath(){

        if (this.currentBranch.get(this.currentBranch.size()-1).state == 3){
            
            for (SearchNode n : this.currentBranch.subList(1, this.currentBranch.size()-1)){
                this.finalRoute[n.x][n.y] = 8;
            }
        }
    }

    //Core function for the depthLimited Search
    public boolean depthLimitedSearch(int limit, boolean printStatus){

        //Set all variables to the initial state
        boolean status = false;
        int searchLevel = 0;
        this.nodeCount = 0;
        this.frontier.clear();
        this.frontier.put(0, new ArrayList<SearchNode>(Arrays.asList(this.startNode)));
        this.scanned.clear();
        this.currentBranch.clear();
        this.explored = new int[this.world.length][this.world[0].length];


        //Auxiliary list of all nodes in the frontier
        List<SearchNode> auxList = new ArrayList<SearchNode>(Arrays.asList(this.startNode));

        //Direction of the search needs to be tracked,
        //i.e. if the road is being explored towards S
        //then this route should be continued while possible
        String direction = "";

        //loop until the limit is reached or there is no more nodes to be explored
        while (searchLevel <= limit || auxList.size()>0){

            //Get one level down in case limit has been surpassed
            if (searchLevel > limit){
                searchLevel--;
            }

            // select node from the frontier, follow the direction if possible
            SearchNode testNode = null;
            while (testNode == null){
                //check if at this level there is any node in the frontier
                if (this.frontier.get(searchLevel).size()>0){
                    //try to get node of the direction that is being taken
                    if (this.frontier.keySet().size()>1){
                        //try to get node in the same direction
                        for (SearchNode n : this.frontier.get(searchLevel)){
                            if (checkDirection(this.currentBranch.get(this.currentBranch.size()-1), n) == direction){
                                testNode = n;
                            }
                        }
                        //if not found take first from the list
                        if (testNode==null){
                            testNode = this.frontier.get(searchLevel).get(0);
                            //update direction
                            direction = checkDirection(this.currentBranch.get(this.currentBranch.size()-1), testNode);
                        }
                    }
                    //if only one node then just take it
                    else{
                        testNode = this.frontier.get(searchLevel).get(0);
                    }
                }
                //if no nodes at the current searchLevel then get down
                else{
                    this.frontier.remove(searchLevel);
                    searchLevel--;
                    //get back to the point where there are nodes in the frontier
                    while (this.frontier.get(searchLevel).size()==0){
                        searchLevel--;
                        //if all levels have been scanned return status
                        if (searchLevel<0){
                            return status;
                        }
                    }
                }
            }
            // increase counter
            this.nodeCount++;

            //add to the explored list with search level
            if (!this.scanned.keySet().contains(searchLevel)){
                this.scanned.put(searchLevel, new ArrayList<SearchNode>(Arrays.asList(testNode)));
            }
            else{
                this.scanned.get(searchLevel).add(testNode);
            }
            
            //if expanding the branch -> add node
            if (searchLevel > this.currentBranch.size()-1){
                this.currentBranch.add(testNode);
            }
            //if retrieving, update branch and then add the test node
            else{
                this.currentBranch = this.currentBranch.subList(0, searchLevel);
                this.currentBranch.add(testNode);
            }
            if (printStatus){
                System.out.println("Current Branch:\n"+this.currentBranch);
            }
            
            //Update the auxiliary list
            ArrayList<SearchNode> adjAux = new ArrayList<SearchNode>();
            for (int k : this.frontier.keySet()){
                for (SearchNode n : this.frontier.get(k)){
                    adjAux.add(n);
                }
            }
            auxList = adjAux;

            if (printStatus){
                System.out.println("\nRunning checks for depth: " + searchLevel + "\n");
                System.out.println("Test Node:\n" + testNode);
            }
        
            //test if there is traffic, then remove node from frontier, 
            //mark as explored and move forward
            if (testNode.state == this.traffic){
                //Mark as explored
                this.explored[testNode.x][testNode.y] = this.traffic;
                //remove from frontier
                this.frontier.get(searchLevel).remove(testNode);
                auxList.remove(testNode);
            }
            //in other cases (0,2) then try to expand frontier 
            else if (testNode.state == 0 || testNode.state == this.start){
                //expand neighbours
                ArrayList<SearchNode> expansion = testNode.expand(this.world);
                //store for nodes that comply with requirements
                ArrayList<SearchNode> toFrontier = new ArrayList<SearchNode>();

                for (SearchNode n : expansion){
                    //chek if the node isnt already in the branch nor in the frontier
                    if (!this.checkContains(this.currentBranch, n) &&
                             (this.frontier.keySet().contains(searchLevel+1) && !this.checkContains(this.frontier.get(searchLevel+1), n) ||
                                !this.frontier.keySet().contains(searchLevel+1)) &&
                                !this.checkExplored(n, searchLevel)){
                        //if traffic on the node then just mark in the explored world matrix
                        if (n.state == this.traffic){
                            this.explored[n.x][n.y] = this.traffic;
                        }
                        //if goal state then update objects and return status
                        else if (n.state == this.goal){
                            this.explored[n.x][n.y]=this.goal;
                            if (testNode.state !=this.start){
                                this.explored[testNode.x][testNode.y] = 8;
                            }
                            else{
                                this.explored[testNode.x][testNode.y] = this.start;
                            }
                            if (printStatus){
                                System.out.println("Goal has been reached");
                            }
                            this.currentBranch.add(n);
                            status = true;
                            this.drawPath();
                            return status;
                        }
                        else{
                            toFrontier.add(n);
                            auxList.add(n);
                        }
                    }
                }
                //check if any node can be placed into the frontier
                //if not then skip
                if (toFrontier.size()!=0){
                    if (this.frontier.containsKey(searchLevel+1)){
                        for (SearchNode n : toFrontier){
                            this.frontier.get(searchLevel+1).add(n);
                        }
                    }
                    else{
                        this.frontier.put(searchLevel+1, toFrontier);
                    }
                    this.frontier.get(searchLevel).remove(testNode);
                    auxList.remove(testNode);
                    searchLevel++;
                }
                else{
                    this.frontier.get(searchLevel).remove(testNode);
                }
                if (testNode.state == this.start){
                    this.explored[testNode.x][testNode.y] = this.start;
                }
                else {
                    this.explored[testNode.x][testNode.y] = 8;
                }

            }
            else if (testNode.state == this.goal){
                this.explored[testNode.x][testNode.y] = this.goal;
                status = true;
                if (printStatus){
                    System.out.println("Goal has been reached");
                }
            }
        }
        return status;
    }

    public boolean iterativeDeepening(){
        //assuming the maximum path cannot be greater than .66 of the world
        int maxRoute = (this.world.length * this.world[0].length)*2/3;
        boolean result = false;
        //tracking execution time
        long totalTime = System.currentTimeMillis();

        for (int i = 1; i <= maxRoute; i++){
            //tracking time of each iteration
            long iterTime = System.currentTimeMillis();
            
            result = this.depthLimitedSearch(i, false);
            
            if (result){
                System.out.println("Goal has been found at level: "+ i);
                System.out.println("\nFINAL ROUTE\n");
                for (int[] r : this.finalRoute){
                    for (int val : r){
                        System.out.print(val + " ");
                    }
                    System.out.println();
                }
                System.out.println("Total number of nodes verified: "+this.nodeCount + "\n");
                long iterStop = System.currentTimeMillis();
                System.out.println("Search with limit: "+ i + " lasted: "+(iterStop-iterTime)/1000 + " seconds\n");
                long totalStop = System.currentTimeMillis();
                System.out.println("Total Elapsed Time: "+(totalStop-totalTime)/1000 + " seconds\n");
                return result;
            }
            else{
                System.out.println("\nUnsuccessful search at level: "+i);
                System.out.println("\nEXPLORED:\n");
                for (int[] r : this.explored){
                    for (int val : r){
                        System.out.print(val + " ");
                    }
                    System.out.println();
                }
            }
            long iterStop = System.currentTimeMillis();
            System.out.println("Search with limit: "+ i + " lasted: "+(iterStop-iterTime)/1000 + " seconds\n");
        }
        System.out.println("Goal has not been found\nLast limit: "+maxRoute);
        long totalStop = System.currentTimeMillis();
        System.out.println("Total Elapsed Time: "+(totalStop-totalTime)/1000 + " seconds\n");
        return result;
    }

    public static void main(String[] args){

        IDSearch test = new IDSearch("IDSearch/grids_test.txt", 2, 3, 1);
        test.iterativeDeepening();
    }
}

class SearchNode {

    // Node's variables: location and depth for the search
    public int x;
    public int y;
    public int depth;
    public int state;

    public SearchNode(int x_in, int y_in, int depth_in, int state_in){
        this.x = x_in;
        this.y = y_in;
        this.depth = depth_in;
        this.state = state_in;
    }

    public boolean checkEqualState(SearchNode node){
        return this.state == node.state;
    }

    public boolean checkEqualNodes(SearchNode node){
        return this.x == node.x && this.y == node.y;
    }

    public ArrayList<SearchNode> expand(int[][] grid){
        // Expand nodes located on N,S, W, E of the current node
        ArrayList<SearchNode> store = new ArrayList<SearchNode>();
        Integer [] horizontal = {this.x -1, this.x +1};
        Integer [] vertical = {this.y -1, this.y +1};
        for (Integer h : horizontal ){
            if (h >= 0 & h <= grid.length-1) {
                store.add(new SearchNode(h, this.y, this.depth + 1, grid[h][this.y]));
            }
        }
        for (Integer v : vertical){
            if (v >= 0 & v <= grid[0].length-1){
                store.add(new SearchNode(this.x, v, this.depth + 1, grid[this.x][v]));
            }
        }
        return store;
    }

    public String toString(){
        return "\nNode's Details:\n"
        + "Location: (" + String.valueOf(this.x) + ", "+ String.valueOf(this.y) + ")\n"
        + "Depth: " + String.valueOf(this.depth);
    }

}



