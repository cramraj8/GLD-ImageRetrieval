import faiss
import os
import pandas as pd
import numpy 
import glob
import numpy as np
from skimage import measure, transform, feature
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ml_metrics as metrics

# DB details
NPZ_FEAT_SRC = "/local/cs572/rchan31/codespace/BaselinePrototype/npz_file_saves/COMPACT_NPZ_DELF_FEATS/"
IMG_SRC_PATH = "/local/cs572/cziems/index/"

DB_LOOKUP_CSV = "/local/cs572/rchan31/fINALwEEK/csv_lookups/FAISS_delf_v1_CSV_lookup.csv"
N_INDEX = 952133
TOP_K = 100



# Query details
Q_NPZ_FEAT_SRC = "/local/cs572/rchan31/fINALwEEK/COMPACT_NPZ_DELF_test_FEATS"
Q_IMG_SRC_PATH = "/local/cs572/cziems/test/"

Q_LOOKUP_CSV = "/local/cs572/rchan31/fINALwEEK/csv_lookups/processed_retrieval_solutions.csv"
Q_N_INDEX = 667

INITIAL_RANKED_LIST_CSV = "/local/cs572/rchan31/fINALwEEK/csv_lookups/initial_retrievals_delf_v1.csv"

RERANKED_LIST_CSV = "/local/cs572/rchan31/fINALwEEK/csv_lookups/reranked_results_FOLDER/reranked_retrievals_delf_v1.csv"


def main_run():

	# Load Query Reference Table
	query_df = pd.read_csv(Q_LOOKUP_CSV)
	query_image_IDs = query_df["id"]

	# Load Database Reference Table
	db_df = pd.read_csv(DB_LOOKUP_CSV)

	# Load Initial Ranked List Table
	init_rank_df = pd.read_csv(INITIAL_RANKED_LIST_CSV)
	init_rank_df.drop(init_rank_df.columns[0], axis=1, inplace=True)
	init_rank_df.set_index("queryImageID", drop=True, inplace = True)


	def rerank_retrievals(q_idx, query_id, ranked_list): # per query
    
	    # =================== reranking part ===================
	    query_npz_path = os.path.join(Q_NPZ_FEAT_SRC, "%s.npz" % query_id)
	    
	    q_locations_x = np.asarray(np.load(query_npz_path)["locations_x"], np.int32)
	    q_locations_y = np.asarray(np.load(query_npz_path)["locations_y"], np.int32)
	    q_locations = np.asarray([q_locations_x, q_locations_y]).T
	    
	    num_features_q = q_locations.shape[0]
	    # ======================================================
	    
	    query_db_inliers = {}
	    db_ret_img_IDs = []
	    return_list = []
	    for topk_idx, ranked_fileID in enumerate(ranked_list):
	        
	        if ranked_fileID == None: # taking care of -1 retrieved images
	            continue
	        
	        # =================== reranking part ===================
	        db_npz_filename_path = os.path.join(NPZ_FEAT_SRC, "%s.npz" % ranked_fileID)
	        insta_imageID = "%s.jpg" % ranked_fileID
	        npz_data = np.load(db_npz_filename_path)
	        
	        inst_location_x = np.asarray(npz_data["locations_x"], np.int32)
	        inst_location_y = np.asarray(npz_data["locations_y"], np.int32)
	        db_locations = np.asarray([inst_location_x, inst_location_y]).T

	        num_features_db = db_locations.shape[0]
	        
	        min_feat_len = min(num_features_q, num_features_db)


	        # Perform geometric verification using RANSAC.
	        _, inliers = measure.ransac((db_locations[:min_feat_len, :],
	                                     q_locations[:min_feat_len, :]),
	                                      transform.AffineTransform,
	                                      min_samples=3,
	                                      residual_threshold=20,
	                                      max_trials=1000)

	        num_inliers = 0
	        if inliers is not None:
	            num_inliers = sum(inliers)
	        print('Found %d inliers for image : %s.jpg' % (num_inliers, ranked_fileID))
	        query_db_inliers[ranked_fileID] = num_inliers
	        db_ret_img_IDs.append(ranked_fileID)
	        # ======================================================
	        
	        # if topk_idx == 2:
                #    break
    
	    # =================== reranking part ===================
	    db_ret_img_IDs.sort(key=query_db_inliers.get, reverse = True)  
	    # ======================================================
	    
	    # if len(db_ret_img_IDs) != TOP_K:
	    #     db_ret_img_IDs = db_ret_img_IDs + ['None'] * (TOP_K-len(db_ret_img_IDs))
        
	    return_list = ["%d-%s" % (val, i) for i, val in zip(db_ret_img_IDs, sorted(query_db_inliers.values(), reverse=True))]
	    if len(return_list) != TOP_K:
	        return_list = return_list + ['None'] * (TOP_K-len(return_list))
	        
	    return return_list


	re_ranked_q_db_pairs = []
	for q_idx, query_row in query_df.iterrows():
	    
	    db_ret_img_IDs = rerank_retrievals(q_idx, query_row["id"], init_rank_df.loc[query_row["id"]])
	    
	    re_ranked_q_db_pairs.append(db_ret_img_IDs)
	    
	    print("............ Finished retrieving for query : %s.jpg : %d / %d" % (query_row["id"], (q_idx + 1), query_df.shape[0]))
	    
	    
	    if q_idx % 2 == 0:
	        reranked_retrieval_df = pd.DataFrame(re_ranked_q_db_pairs)
	        reranked_retrieval_df.insert(0, "queryImageID", query_df["id"].loc[:q_idx + 1])
	        reranked_retrieval_df.to_csv(RERANKED_LIST_CSV)
	        


if __name__ == "__main__":
	main_run()





