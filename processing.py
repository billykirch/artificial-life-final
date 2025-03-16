import subprocess
import random
import json
import tempfile
import copy
from pprint import pprint

max_cilia = 20 # prevent non-existent, too-thin cilia
max_height = 0.15 # discourage optimization for falling forward
min_height = 0.01
h_mutation_strength = 0.01
n_mutation_strength = 1

# Returns parameters for a new ciliate based on input
def mutatedCiliate(ciliateShape):
    height = ciliateShape[0][1] # access the height of the body
    n = len(ciliateShape) - 1

    # mutate the parameters
    m_height = height + random.randint(-1, 1) * h_mutation_strength
    m_n = n + random.randint(-1, 1) * n_mutation_strength

    # bound mutations to reasonable values
    if m_height > max_height:
        m_height = max_height
    elif m_height < min_height:
        m_height = min_height
    
    if m_n > max_cilia:
        m_n = max_cilia
    elif m_n < 1:
        m_n = 1
    
    return createCiliate(m_height, m_n)

# Returns a ciliate shape according to input height and number of cilia
def createCiliate(height, n: int):
    assert n > 0 and n <= 20 # limit number of cilia

    # ciliate length is fixed at 0.3 (prevents evolution toward a longer body)
    length = 0.3
    min_division = 0.01

    division = max(min_division, length / n) # lower limit for cilia width

    shape = []
    shape.append([0.0, height, 0.3, 0.06, -1]) # standard body at adjusted height

    for i in range(n):
        x = division * i
        w = division * 0.75 # create gaps between cilia
        shape.append([x, 0.0, w, height, i])
    
    return shape

# MAIN ----

# creating the first generation of ciliates before mutation:
pop = 7
# vals = random.sample(range(2, 14), pop*2) # random ints from which to draw
shapes = []
for i in range(pop):
    # cilia = random.randint(2, 14)
    # height = random.randint(2, 14) * 0.01
    shapes.append(createCiliate(random.randint(2, 14) * 0.01, random.randint(2, 14)))

# initialize the array that will store all (ciliate, performance) thru all gens
generations = []

num_generations = 10 # number of generations of evolution
num_ciliates = len(shapes)  # number of individuals in the population (hardcoded above) 

for g in range(num_generations):
    
    generation = [] # array for storing ciliates/performances from this generation
    print(f"GENERATION {g}...")
    
    for i in range(num_ciliates):

        # Build JSON array for passage to evaluate.py:
        shapes_json = json.dumps(shapes[i])

        # Open a temporary file and get the filename:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            output_filename = tmp_file.name

        # Run the Taichi subprocess to evaluate the ciliate:
        print(f"Starting GEN #{g} CILIATE #{i} with shapes: {shapes[i]}")    
        subprocess.run(
            ["python", "evaluate.py", "--shapes", shapes_json, "--gen", str(g), "--ciliate", str(i), "--output", output_filename]
        )

        # Pull the performance value from the temporary file:
        with open(output_filename, "r") as f:
            output = json.load(f)
        
        # Save the performance value to the generation array as a ciliate object
        ciliate = {"ciliate" : shapes[i], "perf": output["perf"], "id": i, "gen": g}
        generation.append(ciliate)
            
    # Add the creaciliatetures from this generation to the generations array:
    generations.append(generation) # append ciliates in this generation to overall results array

    # Sort the results by performance, ascending:
    sorted_generation = sorted(generation, key=lambda x: x["perf"]) # most negative values at beginning

    # Slice off the worst performing ciliate, duplicate the best one:
    selected_generation = sorted_generation[:-1]
    selected_generation.insert(0, copy.deepcopy(selected_generation[0])) # insert copy of best at front
    assert len(selected_generation) == num_ciliates

    # mutate ciliates:
    shapes = [
        # exclude one copy of best ciliate (continue testing for beneficial mutation on best ciliate)
        mutatedCiliate(x["ciliate"]) if i > 0 else x["ciliate"]
        for i, x in enumerate(selected_generation)
    ]
print("GENERATIONS ARRAY:")
pprint(generations)