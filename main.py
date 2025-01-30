import copy

# Basic data structure, which can nest to represent math equations
class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

# convert string representation into tree
def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root") # add a dummy node
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ') # count the spaces, which is crucial information in a string representation
        node_name = line.strip() # remove spaces, when putting it in the tree form
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0] # remove dummy node

# convert tree into string representation
def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name) # spacings
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1) # one node in one line
        return result
    return recursive_str(node)

# Generate transformations of a given equation provided only one formula to do so
# We can call this function multiple times with different formulas, in case we want to use more than one
# This function is also responsible for computing arithmetic, pass do_only_arithmetic as True (others param it would ignore), to do so
def apply_individual_formula_on_given_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic=False, structure_satisfy=False):
    variable_list = {}
    
    def node_type(s):
        if s[:2] == "f_":
            return s
        else:
            return s[:2]
    def does_given_equation_satisfy_forumla_lhs_structure(equation, formula_lhs):
        nonlocal variable_list
        # u can accept anything and p is expecting only integers
        # if there is variable in the formula
        if node_type(formula_lhs.name) in {"u_", "p_"}: 
            if formula_lhs.name in variable_list.keys(): # check if that variable has previously appeared or not
                return str_form(variable_list[formula_lhs.name]) == str_form(equation) # if yes, then the contents should be same
            else: # otherwise, extract the data from the given equation
                if node_type(formula_lhs.name) == "p_" and "v_" in str_form(equation): # if formula has a p type variable, it only accepts integers
                    return False
                variable_list[formula_lhs.name] = copy.deepcopy(equation)
                return True
        if equation.name != formula_lhs.name or len(equation.children) != len(formula_lhs.children): # the formula structure should match with given equation
            return False
        for i in range(len(equation.children)): # go through every children and explore the whole formula / equation
            if does_given_equation_satisfy_forumla_lhs_structure(equation.children[i], formula_lhs.children[i]) is False:
                return False
        return True
    if structure_satisfy:
      return does_given_equation_satisfy_forumla_lhs_structure(equation, formula_lhs)
    # transform the equation as a whole aka perform the transformation operation on the entire thing and not only on a certain part of the equation
    def formula_apply_root(formula):
        nonlocal variable_list
        if formula.name in variable_list.keys():
            return variable_list[formula.name] # fill the extracted data on the formula rhs structure
        data_to_return = TreeNode(formula.name, None) # produce nodes for the new transformed equation
        for child in formula.children:
            data_to_return.children.append(formula_apply_root(copy.deepcopy(child))) # slowly build the transformed equation
        return data_to_return
    count_target_node = 1
    # try applying formula on various parts of the equation
    def formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic):
        nonlocal variable_list
        nonlocal count_target_node
        data_to_return = TreeNode(equation.name, children=[])
        variable_list = {}
        if do_only_arithmetic == False:
            if does_given_equation_satisfy_forumla_lhs_structure(equation, copy.deepcopy(formula_lhs)) is True: # if formula lhs structure is satisfied by the equation given
                count_target_node -= 1
                if count_target_node == 0: # and its the location we want to do the transformation on
                    return formula_apply_root(copy.deepcopy(formula_rhs)) # transform
        else: # perform arithmetic
            if len(equation.children) == 2 and all(node_type(item.name) == "d_" for item in equation.children): # if only numbers
                x = []
                for item in equation.children:
                    x.append(int(item.name[2:])) # convert string into a number
                if equation.name == "f_add":
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        return TreeNode("d_" + str(sum(x))) # add all
                elif equation.name == "f_mul":
                    count_target_node -= 1
                    if count_target_node == 0:
                        p = 1
                        for item in x:
                            p *= item # multiply all
                        return TreeNode("d_" + str(p))
                elif equation.name == "f_pow" and x[1]>=2: # power should be two or a natural number more than two
                    count_target_node -= 1
                    if count_target_node == 0:
                        return TreeNode("d_"+str(int(x[0]**x[1])))
                elif equation.name == "f_sub":
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        return TreeNode("d_" + str(x[1]-x[0]))
                elif equation.name == "f_div" and int(x[0]/x[1]) == x[0]/x[1]:
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        return TreeNode("d_" + str(int(x[0]/x[1])))
        if node_type(equation.name) in {"d_", "v_"}: # reached a leaf node
            return equation
        for child in equation.children: # slowly build the transformed equation
            data_to_return.children.append(formula_apply_various_sub_equation(copy.deepcopy(child), formula_lhs, formula_rhs, do_only_arithmetic))
        return data_to_return
    cn = 0
    # count how many locations are present in the given equation
    def count_nodes(equation):
        nonlocal cn
        cn += 1
        for child in equation.children:
            count_nodes(child)
    transformed_equation_list = []
    count_nodes(equation)
    for i in range(1, cn + 1): # iterate over all location in the equation tree
        count_target_node = i
        orig_len = len(transformed_equation_list)
        tmp = formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic)
        if str_form(tmp) != str_form(equation): # don't produce duplication, or don't if nothing changed because of transformation impossbility in that location
            transformed_equation_list.append(str_form(tmp)) # add this transformation to our list
    return transformed_equation_list 

# Function to generate neighbor equations
def generate_transformation(equation, file_name):
    input_f, output_f = return_formula_file(file_name) # load formula file
    transformed_equation_list = []
    for i in range(len(input_f)): # go through all formulas and collect if they can possibly transform
        transformed_equation_list += [(i, x) for x in apply_individual_formula_on_given_equation(tree_form(copy.deepcopy(equation)), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))]
    return list(set(transformed_equation_list)) # set list to remove duplications

# Function to generate neighbor equations
def generate_arithmetical_transformation(equation):
    transformed_equation_list = []
    transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), None, None, True) # perform arithmetic
    return list(set(transformed_equation_list)) # set list to remove duplications

def string_equation_helper(equation_tree):
    if equation_tree.children == []:
        return equation_tree.name # leaf node
    s = "(" # bracket
    if len(equation_tree.children) == 1:
        s = equation_tree.name[2:] + s
    sign = {"f_add": "+", "f_mul": "*", "f_pow": "^", "f_div": "/", "f_int": ",", "f_sub": "-", "f_dif": "?", "f_int": "?", "f_sin": "?", "f_cos": "?", "f_tan": "?", "f_eq": "=", "f_sqt": "?"} # operation symbols
    for child in equation_tree.children:
        s+= string_equation_helper(copy.deepcopy(child)) + sign[equation_tree.name]
    s = s[:-1] + ")"
    return s

# fancy print main function
def string_equation(eq):
    eq = eq.replace("u_","v_")
    eq = eq.replace("v_0", "x")
    eq = eq.replace("v_1", "y")
    eq = eq.replace("v_2", "z")
    eq = eq.replace("d_", "")
    
    return string_equation_helper(tree_form(eq))

# Function to read formula file
def return_formula_file(file_name):
    with open(file_name, 'r') as file:
      content = file.read()
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)] # alternative formula lhs and then formula rhs
    output_f = [x[i] for i in range(1, len(x), 2)]
    input_f = [tree_form(item) for item in input_f] # convert into tree form
    output_f = [tree_form(item) for item in output_f]
    #for i in range(len(input_f)):
    #  print(string_equation(str_form(input_f[i])), "=", string_equation(str_form(output_f[i])))
    return [input_f, output_f] # return

def search(equation, depth, file_list, auto_arithmetic=True, visited=None):
    if depth == 0: # limit the search
        return None
    if visited is None:
        visited = set()

    print(string_equation(equation))
    if equation in visited:
        return None
    visited.add(equation)
    output =[]
    if file_list[0]:
      output += generate_transformation(equation, file_list[0])
    if auto_arithmetic:
      output += generate_arithmetical_transformation(equation)
    if len(output) > 0:
      output = [output[0]]
    else:
      if file_list[1]:
        output += generate_transformation(equation, file_list[1])
      if not auto_arithmetic:
        output += generate_arithmetical_transformation(equation)
      if file_list[2] and len(output) == 0:
          output += generate_transformation(equation, file_list[2])
    for i in range(len(output)):
        result = search(output[i], depth-1, file_list, auto_arithmetic, visited) # recursively find even more equals
        if result is not None:
            output += result # hoard them
    output = list(set(output))
    return output

# fancy print


def replace(equation, find, r):
  if str_form(equation) == str_form(find):
    return r
  col = TreeNode(equation.name, [])
  for child in equation.children:
    col.children.append(replace(child, find, r))
  return col


import random

GAMMA = 0.9  # Discount factor
ALPHA = 0.1  # Learning rate
EPSILON = 0.5  # Exploration rate
REWARD_VALUE = 10  # Reward for solution
PENALTY_VALUE = -5  # Penalty for exceeding depth
VISIT_LIMIT = 3  # Max times a node can be revisited
MAX_SEARCH_NODES = 250  # Max nodes to search during solving
MAX_DEPTH = 25  # Maximum depth for DFS training

class RL_DFS_Agent:
    def __init__(self, edge_generator):
        self.edge_generator = edge_generator  # Function to generate edges and their types
        self.q_table = {}  # Q-values: {state: {action: value}}
        self.visit_count = {}  # Track node visits
    
    def get_q_values(self, state):
        """Initialize Q-values for unseen states."""
        if state not in self.q_table:
            self.q_table[state] = {edge_type: 0 for _, edge_type in self.edge_generator(state)}  # Initialize for each edge type
        return self.q_table[state]

    def choose_action(self, state):
        """Choose an edge type for a given state using epsilon-greedy."""
        q_values = self.get_q_values(state)
        
        if not q_values:  # If no actions exist for the state
            return None  # No action to take; signal the end of the traversal

        # Epsilon-greedy strategy: exploration vs. exploitation
        if random.random() < EPSILON:
            return random.choice(list(q_values.keys()))  # Explore (random action)
        
        return max(q_values, key=q_values.get)  # Exploit (best action)

    def update_q_value(self, state, action, reward, next_state):
        """Q-Learning Update Rule (Fixed: Ensure Q-values exist)."""
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)  # Ensure next state is initialized
        best_future_q = max(next_q_values.values(), default=0)

        # Fix: Ensure action exists in q_values before updating
        if action not in q_values:
            q_values[action] = 0  # Initialize new action

        q_values[action] += ALPHA * (reward + GAMMA * best_future_q - q_values[action])

    def dfs_train(self, start_node, episodes=1000):
        """Train the agent by running DFS with Q-learning."""
        self.count = 0
        for _ in range(episodes):
            nodenum =0
            stack = [(start_node, 0)]  # Stack for DFS: (node, depth)
            visited = set()
            self.visit_count = {}  # Reset visit counts

            while stack:
                node, depth = stack.pop()  # Pop from the end of the stack (LIFO order)
                
                if depth > MAX_DEPTH:  # Stop if depth exceeds MAX_DEPTH
                    continue

                if self.visit_count.get(node, 0) >= VISIT_LIMIT:
                    continue  # Ignore over-visited nodes

                visited.add(node)
                self.visit_count[node] = self.visit_count.get(node, 0) + 1  # Track visits

                # Determine reward or penalty
                if is_found(node):  # Use is_found to check for solution
                    reward = REWARD_VALUE  # Found solution
                    next_state = node  # Terminal state
                    self.count += 1
                    break  # Stop the DFS search immediately when solution is found
                elif nodenum > MAX_SEARCH_NODES:
                    reward = PENALTY_VALUE
                    next_state = node  # Terminal state
                    break  # Stop the DFS search immediately when solution is found
                else:
                    reward = 0.0  # Small step cost to encourage exploration
                    next_state = node  # Next move
                nodenum += 1
                # Ensure an action is always chosen (fallback if no valid actions)
                q_values = self.get_q_values(node)
                if not q_values:  # If no Q-values exist, initialize them
                    q_values = {edge_type: 0 for _, edge_type in self.edge_generator(node)}

                if not q_values:  # If no valid actions, terminate the DFS (end the traversal)
                    break  # End the current DFS path since no valid moves are available

                # Choose the action (edge type)
                action = self.choose_action(node)
                if action is None:  # No valid action found, terminate the search
                    break  # End traversal if no action is chosen (dead-end)

                # Q-learning update
                self.update_q_value(node, action, reward, next_state)

                # Add next action to the stack if it’s valid
                if not is_found(node) and depth < MAX_DEPTH:
                    stack.append((action, depth + 1))  # Continue exploring until found

    def dfs_solve(self, start_node):
        """Use trained Q-values to perform DFS."""
        stack = [(start_node, 0)]  # Use a stack for DFS: (node, depth)
        visited = set()
        path = []
        nodes_searched = 0  # Counter for nodes searched

        while stack:
            node, depth = stack.pop()  # Pop from the end of the stack (LIFO order)
            nodes_searched += 1  # Increment node search count

            if node in visited:
                continue
            visited.add(node)
            path.append(node)

            if is_found(node):  # Check if node satisfies the solution criteria
                return path, nodes_searched  # Immediately return the path when solution is found

            if nodes_searched >= MAX_SEARCH_NODES:  # Stop if we exceed the max search limit
                return None  # Stop the search and return None

            # Continue the search by exploring deeper paths
            next_move = self.choose_action(node)
            if next_move and next_move not in visited:  # Prevent loops
                stack.append((next_move, depth + 1))  # Add to stack for further exploration

        return None  # Return None if no solution is found



# Edge Generating Function (Using Edge Types as Tuples)
def generate_edges(node):
    """Generates edge types dynamically based on node. Returns a list of tuples (edge, edge_type)."""
    transformations = generate_transformation(node, "formula_list.txt")
    arithmetic = generate_arithmetical_transformation(node)
    
    # Combining and creating tuples (edge, edge_type)
    edges = transformations + [("arithmetic", edge) for edge in arithmetic]
    
    return edges


def is_found(node):
    """Check if the node matches the solution criteria."""
    return "f_int" not in node and "f_dif" not in node  # Example condition


# Example equations
eq6 = """f_int
 f_mul
  f_add
   d_1
   f_mul
    d_2
    f_add
     f_add
      v_0
      d_1
     v_0
  f_dif
   v_0"""
eq5 = """f_int
 f_mul
  f_mul
   d_2
   f_add
    f_add
     v_0
     d_1
    v_0
  f_dif
   v_0"""
eq1 = """f_int
 f_mul
  f_mul
   d_2
   f_add
    v_0
    v_0
  f_dif
   v_0"""
eq2 = """f_int
 f_mul
  f_mul
   d_7
   f_add
    v_0
    v_0
  f_dif
   v_0"""
eq3 = """f_int
 f_mul
  f_add
   v_0
   d_4
  f_dif
   v_0"""
eq4 = """f_int
 f_mul
  f_add
   v_0
   v_0
  f_dif
   v_0"""
eq_list = [eq4, eq3, eq2]
print('started')

# Train the RL Agent with dynamic edges (edge types)
agent = RL_DFS_Agent(generate_edges)  # Use RL_DFS_Agent for training with DFS
for index, eq in enumerate(eq_list):
    agent.dfs_train(start_node=eq, episodes=100)  # Training with DFS
    print(index+1, "/", len(eq_list), ",", agent.count, "/", 100)
# Solve using the trained agent
print("\nSolving using RL-guided DFS...")
while True:
    tmp = agent.dfs_solve(start_node=eq1)  # Solving with BFS
    if tmp is not None:
        for node in tmp[0]:
            print(string_equation(node))  # Print solution path element by element
        break
