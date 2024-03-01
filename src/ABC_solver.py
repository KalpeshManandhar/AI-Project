import numpy as np;

class ABC_solver:
    def __init__(self, limit, n_sources, n_params,
                 sol_range_low, sol_range_high,
                 max_iterations = 50000, n_onlookers = None) -> None:
        # control parameters
        self.LIMIT          = limit
        self.N_SOURCES      = n_sources
        self.N_PARAMS       = n_params
        self.N_ONLOOKERS    = n_sources if (n_onlookers == None) else n_onlookers  
        self.MAX_ITERATIONS = max_iterations
        
        self.solution_range = (sol_range_low, sol_range_high)
        
        self.bestSolutionIndex = 0
        self.bestSolutionFit = 0

        # rng 
        self.rng = np.random.default_rng()

        # sources and no of visits to each
        self.food_sources = []
        self.no_of_visits = np.zeros(self.N_SOURCES)
        for i in range(0, self.N_SOURCES):
            self.food_sources.append(self.get_random_food_source())


        

    
    # set the function used to calculate the fit value of a solution
    # (amount of nectar of the source)
    def setFitFunction(self, fitFunc):
        self.fitFunc = fitFunc


    # equivalent to having a scout bee find a new source
    # Generates random values between the solution range
    def get_random_food_source(self) -> np.array:
        return self.rng.random(self.N_PARAMS) * (self.solution_range[1] - self.solution_range[0]) + self.solution_range[0]

    
    # search for better solution in local space of given solution
    def try_find_better_source(self, current_food_source) -> np.array:
        # get a random food source from the list
        #Returns an intex ranging from 0 to N_sources
        random_food_source_index = self.rng.integers(self.N_SOURCES, size=1)[0]         
        random_food_source = self.food_sources[random_food_source_index]

        # If the current_food_source is equal to random_food_source generated, choose another random partner
        while np.array_equal(random_food_source, current_food_source):
            random_food_source_index = self.rng.integers(self.N_SOURCES, size=1)[0]
            random_food_source = self.food_sources[random_food_source_index]        
        
        # a random value in range [-1,1]
        multiplier = 2 * self.rng.random(1) - 1

        # try to find a new source better than the current one
        new_food_source = current_food_source + multiplier * random_food_source
        
        new_food_source = np.maximum(new_food_source, self.solution_range[0])
        new_food_source = np.minimum(new_food_source, self.solution_range[1])
        return new_food_source

    
    def choose_source(self) -> int:
        # fit value HIGH -> probability HIGH
        # Applying fitFuc rowWise
        fit_values = np.apply_along_axis(self.fitFunc, 1, self.food_sources)

        sum_fit = fit_values.sum()
        probability_values = fit_values/sum_fit 
        
        # multiply probability with random values 
        random_values = self.rng.random(self.N_SOURCES)
        actual_occurence = probability_values * random_values

        # choose the occurence with max value
        return actual_occurence.argmax()
    

    def send_employed_bees(self) -> None:
        for i in range(0, self.N_SOURCES):
            # find new source near current source
            new_source = self.try_find_better_source(self.food_sources[i])

            new_fit = self.fitFunc(new_source)
            current_fit = self.fitFunc(self.food_sources[i])
            # if new source is better, replace
            if new_fit > current_fit:
                self.food_sources[i] = new_source
                self.no_of_visits[i] = 0
                if new_fit > self.bestSolutionFit:
                    self.bestSolutionFit = new_fit
                    self.bestSolutionIndex = i

            # if not, keep the current source
            else:
                self.no_of_visits[i] += 1
                
                # if number of visits is greater than the limit, abandon and search for random
                if self.no_of_visits[i] > self.LIMIT:
                    self.no_of_visits[i] = 0
                    self.food_sources[i] = self.get_random_food_source()


    def send_onlooker_bees(self) -> None:
        for i in range(0, self.N_ONLOOKERS):
            # choose a food source based on its fit
            chosen_index = self.choose_source()

            # find new source near chosen source
            new_source = self.try_find_better_source(self.food_sources[chosen_index])

            # if new source is better, replace
            if self.fitFunc(new_source) > self.fitFunc(self.food_sources[chosen_index]):
                self.food_sources[chosen_index] = new_source
                self.no_of_visits[chosen_index] = 0
            

    def one_cycle(self):
        self.send_employed_bees()
        self.send_onlooker_bees()


    # returns the solution with the max fit value
    # intermediate values at adds the sources value at each step included
    def solve(self, intermediate_values_at: list = []):
        intermediates = np.zeros((len(intermediate_values_at), self.N_SOURCES, self.N_PARAMS))
        intermediate_index = 0
        add_intermediates = True
        if len(intermediate_values_at) == 0:
            add_intermediates = False

        for i in range(0, self.MAX_ITERATIONS):
            self.one_cycle()
            print(f'Cycle: {i} Best fit: {self.bestSolutionFit}')
            if (add_intermediates and intermediate_values_at.count(i) > 0):
                intermediates[intermediate_index] = np.copy(self.food_sources)
                intermediate_index += 1

        
        fit_values = np.apply_along_axis(self.fitFunc, 1, self.food_sources)
        max_fit_index = fit_values.argmax()

        return (self.food_sources[max_fit_index], intermediates)