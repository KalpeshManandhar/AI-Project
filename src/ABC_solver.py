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
        
        # rng 
        self.rng = np.random.default_rng();

        # sources and no of visits to each
        self.food_sources = []
        self.no_of_visits = np.zeros(self.N_SOURCES)
        for i in range(0, self.N_SOURCES):
            self.food_sources.append(self.get_random_food_source())


        pass

    
    # set the function used to calculate the fit value of a solution
    # (amount of nectar of the source)
    def setFitFunction(self, fitFunc):
        self.fitFunc = fitFunc


    # equivalent to having a scout bee find a new source
    def get_random_food_source(self) -> np.array:
        return self.rng.random(self.N_PARAMS) * (self.solution_range[1] - self.solution_range[0]) + self.solution_range[0]

    
    # search for better solution in local space of given solution
    def try_find_better_source(self, current_food_source) -> np.array:
        # get a random food source from the list
        random_food_source_index = self.rng.integers(self.N_SOURCES, size=1)[0]
        random_food_source = self.food_sources[random_food_source_index]
        
        # a random value in range [-1,1]
        multiplier = 2 * self.rng.random(1) - 1

        # try to find a new source better than the current one
        new_food_source = current_food_source + multiplier * random_food_source

        return new_food_source

    
    def choose_source(self) -> int:
        # fit value HIGH -> probability HIGH
        fit_values = np.apply_along_axis(self.fitFunc, 1, self.food_sources)

        sum_fit = fit_values.sum()
        probability_values = fit_values/sum_fit 
        
        # multiply probability with random values 
        random_values = self.rng.random(self.N_SOURCES);
        actual_occurence = probability_values * random_values

        # choose the occurence with max value
        return actual_occurence.argmax()
    

    def send_employed_bees(self) -> None:
        for i in range(0, self.N_SOURCES):
            # find new source near current source
            new_source = self.try_find_better_source(self.food_sources[i])

            # if new source is better, replace
            if self.fitFunc(new_source) > self.fitFunc(self.food_sources[i]):
                self.food_sources[i] = new_source
                self.no_of_visits[i] = 0
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

        for i in range(0, self.MAX_ITERATIONS):
            self.one_cycle()
            if (intermediate_values_at.count(i) > 0):
                intermediates[intermediate_index] = np.copy(self.food_sources)

        
        fit_values = np.apply_along_axis(self.fitFunc, 1, self.food_sources)
        max_fit_index = fit_values.argmax()

        return (self.food_sources[max_fit_index], intermediates)