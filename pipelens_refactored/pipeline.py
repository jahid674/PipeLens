#read a config file to identify parameters and components!
#Each module exposes a name and parameters
#Define an optimization function that takes pipeline and search space as input to return the best pipeline


#To think: How to allow iterations!
#How to allow addition of new components

class Pipeline:
    def __init__(self, name):
        print ("initializing pipeline")
        self.name = name
        return



if __name__ == "__main__":
    pipe=Pipeline("test pipeline")