Description of the files:
mazes: contains the two mazes in this experiements (taken from RL-MDP's
sample mazes)
    2_medium_simple.maze: the small 10x10 maze
    8_big_maze: the large maze

Experiments to run in RL-MDP:

    Small maze:

        Value iteration: 
            Parameters: PJOG: 0.3, precision: 0.001
        Policy iteration:
            Parameters: PJOG: 0.3, value limit: (run for each of
            these values): [1, 2, 5, 10], iter limit: 5000, precition: 0.001
        Q-learning:
            Parametres: PJOG: 0.3, Epsilon (run for each of
            these values): [0, 0.1, 0.2, 0.5], cycles: 1000, precision: 0.001,
            learning rate: 0.7
    
    Large maze:
        Value iteration: 
            Parameters: PJOG: 0.3, precision: 0.001
        Policy iteration:
            Parameters: PJOG: 0.3, value limit: (run for each of
            these values): [1, 2, 5, 10], iter limit: 5000, precition: 0.001
        Q-learning:
            Parametres: PJOG: 0.3, Epsilon (run for each of
            these values): [0, 0.1, 0.2, 0.5], cycles: 1000, precision: 0.001,
            learning rate: 0.7

Screenshots were taken after these were run (located in the images folder)
paper: contains the tex and pdf file for the maze

Files (images and mazes) can be found here: https://github.com/arvindr9/ml/tree/master/hw4