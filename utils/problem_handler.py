import numpy as np
class ProblemHandler(object):
    def __init__(self, problem_classes, max_num_open_problems=100):
        self.open_problems = {}
        self.following_id = 0
        self.problem_classes = problem_classes
        self.max_num_open_problems = max_num_open_problems

    def open_new_problem(self, problem_class_id=0):
        assert len(
            self.open_problems) < self.max_num_open_problems, "maximal number of open problems reached, cannot open another one!"
        problem_class = self.problem_classes[problem_class_id][0]
        init_kwargs = self.problem_classes[problem_class_id][1]

        new_open_problem = {}
        new_open_problem["problem"] = problem_class(**init_kwargs)
        new_open_problem["step"] = 0
        new_open_problem["last_solution"] = new_open_problem["problem"].initial_condition
        new_open_problem["max_steps"] = new_open_problem["problem"].time_steps

        self.open_problems[self.following_id] = new_open_problem
        self.following_id += 1
        return new_open_problem, self.following_id - 1

    def update_problem(self, open_problem_id, new_solution):
        self.open_problems[open_problem_id]["last_solution"] = new_solution
        self.open_problems[open_problem_id]["step"] += 1
        if self.open_problems[open_problem_id]["step"] >= self.open_problems[open_problem_id]["max_steps"]:
            self.close_problem(open_problem_id)

    def close_problem(self, open_problem_id):
        self.open_problems.pop(open_problem_id)
        print("Problem closed")

    def get_random_problem(self, opening_prob):
        if ((len(self.open_problems) < self.max_num_open_problems) \
            and (np.random.rand() < opening_prob)) \
                or (len(self.open_problems) == 0):
            class_id = np.random.randint(len(self.problem_classes))
            problem, problem_id = self.open_new_problem(problem_class_id=class_id)
            print("New problem opened!")
        else:
            problem_id = np.random.choice(list(self.open_problems.keys()))
            problem = self.open_problems[problem_id]
        return problem, problem_id