from itertools import product, permutations
import numpy as np
# Lists of elements
 
imputation = ["MICE", "EM", "KNN", "MF"]
normalizer_strategies = ['DS','MM','ZS']
feature_selection = ['MR','WR','LC','Tree']
outlier_detection =  ["ZSB", "LOF", "IQR"]
consist_checker =  ["CC", "PC"]
duplicate_detector =  ["ED", "AD"]
class_map = {
    "MICE": 0, "EM": 1, "KNN": 2, "MF": 3,
    "DS": 4, "MM": 5, "ZS": 6,
    "MR": 7, "WR": 8, "LC": 9, "Tree": 10,
    "ZSB": 11, "LOF": 12, "IQR": 13,
    "CC": 14, "PC": 15,
    "ED": 16, "AD": 17,
    "LASSO": 18, "OLS": 19, "MARS": 20,
    "HCA": 21, "KMEANS": 22,
    "CART": 23, "LDA": 24, "NB": 25
}

                        # Generate the Cartesian product of the lists
cartesian_product = list(product( outlier_detection,duplicate_detector,imputation))

# Generate all permutations of each combination in the Cartesian product

r = np.array([
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],
                [-1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 100],

                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0,  0,  0,  0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0,
                 -1],

                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],
                [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, 0,
                 -1],

                [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0,
                 0, 100],
                [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0,
                 0, 100],
                [0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0,
                 0, 100],

                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, 0, 0, -1, -1,
                 0, 0, 100],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0, 0, 0, -1, -1,
                 0, 0, 100],

                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 100],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                 -1, -1, -1, -1]]).astype("float32")

def get_none(value):
        if value in [0,1,2,3,4] :
                return 'None_imp'
        elif value in [5,6,7,8]:
                return 'None_od'
        elif value in [12,13,14]:
                return 'None_duplicate'
        else:
                import pdb;pdb.set_trace()

def remove_duplicates_and_check(lst):
        seen = set()
        result = []
        has_duplicates = False

        for item in lst:
                if item in seen:
                        has_duplicates = True
                else:
                        seen.add(item)
                result.append(item)
        
        return  has_duplicates
def get_valid_moves(traverse_tuple):
        class_map = {
                "MICE": 0, "EM": 1, "KNN": 2, "MF": 3,'None_imp':4,
                "ZSB": 5, "LOF": 6, "IQR": 7,'None_od':8,
                "CC": 9, "PC": 10,"None_cons":11,
                "ED": 12, "AD": 13,'None_duplicate':14,
                'Reg':15,
                "DS":0,"MM":1, "ZS":2,
                "MR":3, "WR":4, "LC":5, "Tree":6,
                }
        

        # print(traverse_tuple)
        list_of_mov = []
        # for i in range(19):
                # current_mov = []
                # traverse_tuple = ('AD','LOF','MICE')
                # a = r[i,class_map[traverse_tuple[0]]]
                # b = r[i,class_map[traverse_tuple[1]]]
                # c = r[i,class_map[traverse_tuple[2]]]

                # if(a>-1):
                #         current_mov.append(traverse_tuple[0])
                # else :
                #         current_mov.append(get_none(class_map[traverse_tuple[0]]))
                

                # if(b>-1):
                #         current_mov.append(traverse_tuple[1])
                # else:
                #         current_mov.append(get_none(class_map[traverse_tuple[0]]))
                
                # if(c>-1):
                #         current_mov.append(traverse_tuple[2])
                # else:
                #         current_mov.append(get_none(class_map[traverse_tuple[0]]))
                # list_of_mov.append(current_mov)

        if r[class_map[traverse_tuple[0]] , class_map[traverse_tuple[1]]] + r[class_map[traverse_tuple[1]], class_map[traverse_tuple[2]]] +r[class_map[traverse_tuple[0]], class_map[traverse_tuple[2]]]==0:
                return [traverse_tuple[0],traverse_tuple[1],traverse_tuple[2]]
                

        
        return []  # All moves were valid

all_combinations_with_permutations = []
for combination in cartesian_product:
        all_combinations_with_permutations.extend(permutations(combination))

valid_moves = []
for val in all_combinations_with_permutations:
        # for filter_tuple in get_valid_moves(val):
        #     if not remove_duplicates_and_check(filter_tuple):
        valid_pipeline = get_valid_moves(val)
        if len(valid_pipeline)>0:
                valid_moves.append([valid_pipeline[0],valid_pipeline[1],valid_pipeline[2]])

# for val in valid_moves:
print(valid_moves)


# def remove_duplicates(arrays):
#     # Convert each list to a tuple so they can be added to a set
#     unique_tuples = set(tuple(array) for array in arrays)
    
#     # Convert tuples back to lists
#     unique_lists = [list(tup) for tup in unique_tuples]
    
#     return unique_lists



# filtered_duplicate_list =remove_duplicates(valid_moves)
# print(filtered_duplicate_list)
# print(len(filtered_duplicate_list))






