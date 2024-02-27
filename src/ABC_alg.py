import numpy as np;
from math import exp;


NO_OF_SOURCES = 20
NO_OF_VARS = 5
LIMIT = 100

rng = np.random.default_rng();

def get_random_food_source(n):
    return rng.random(n) * 100 - 50


# test function
minima = [2,-3,1,0,0]
def utility(food_source):
    food_source = food_source + np.array([-2,3,-1,0,0])
    return np.dot(food_source, food_source)

    return (food_source[0] - 2)**2 + (food_source[1] + 3)**2 + \
        (food_source[2] - 1)**2 + food_source[3]**2 + food_source[4]**2

# is a better than b
def isbetter(source_a, source_b):
    return utility(source_a) < utility(source_b)




food_sources = []
fit_food_sources = np.zeros(NO_OF_SOURCES)
employed_bees = []
onlooker_bees = []

def try_find_better_source(current_food_source) -> np.array:
    # get a random food source from the list
    random_food_source_index = rng.integers(NO_OF_SOURCES, size=1)[0]
    random_food_source = food_sources[random_food_source_index]
    
    # a random value in range [-1,1]
    multiplier = 2 * rng.random(1) - 1

    # try to find a new source better than the current one
    new_food_source = current_food_source + multiplier * random_food_source

    return new_food_source




class EmployedBee:
    def __init__(self, bee_no) -> None:
        self.role = "employed"
        self.current_food_source_index = bee_no
        self.no_of_visits = 0

    def search_better_source(self):
        # current best food source of the employed bee
        current_food_source = food_sources[self.current_food_source_index]

        # get new food source near current source
        new_food_source = try_find_better_source(current_food_source)

        # if current is better then keep the current in memory
        if isbetter(current_food_source, new_food_source):
            self.no_of_visits += 1

            # if number of visits to the same source doesnt find any better sources
            # abandon source 
            # (equivalent to having a scout)
            if self.no_of_visits > LIMIT:
                food_sources[self.current_food_source_index] = get_random_food_source(NO_OF_VARS)
                self.no_of_visits = 0

        # if new source is better, remember new and forget old
        else:
            food_sources[self.current_food_source_index] = new_food_source
            self.no_of_visits = 0


class OnlookerBee:
    def __init__(self) -> None:
        self.role = "onlooker"

    # TODO: implement 
    def choose_source(self) -> int:
        # utility (error) LOW -> fit value HIGH -> probability HIGH
        utility_values = np.apply_along_axis(utility, 1, food_sources)
        fit_values = np.exp(-np.abs(utility_values))

        sum_fit = fit_values.sum()
        probability_values = fit_values/sum_fit 
        
        # multiply probability with random values 
        random_values = rng.random(NO_OF_SOURCES);
        actual_occurence = probability_values * random_values

        # choose the occurence with max value
        return actual_occurence.argmax()
    
    
    def search_better_source(self):
        # choose a source 
        chosen_source_index = self.choose_source()
        chosen_source = food_sources[chosen_source_index]

        # try finding a better source
        new_source = try_find_better_source(chosen_source)

        # if new source is better, replace
        if isbetter(new_source, chosen_source):
            food_sources[chosen_source_index] = new_source




# initialize with random food sources
for i in range(0,NO_OF_SOURCES):
    food_sources.append(get_random_food_source(NO_OF_VARS))
    employed_bees.append(EmployedBee(i))
    onlooker_bees.append(OnlookerBee())



for i in range(0,50000):    
    # all employed bees try finding better sources in the local space of their sources
    for bee in employed_bees:
        bee.search_better_source()

    # all onlooker beese cheese (bees choose) a random source to investigate further
    # choosing is probabilistic based on how good the source is
    # probability is given by fit(source)/sum(fit(sources))
    for bee in onlooker_bees:
        bee.search_better_source()

for source in food_sources:
    print(f'{source} :{utility(source)}')

        



