import os
from pathlib import Path

# pretrain weight save folder
absolute_path = str(Path(__file__).resolve().parent) + os.sep

visualize_path = absolute_path + "visualize/"

# amazon dataset original files
amazon_music = absolute_path + "data/amazon/music/reviews_CDs_and_Vinyl_5.json"
amazon_movie = absolute_path + "data/amazon/movie/reviews_Movies_and_TV_5.json"
amazon_book = absolute_path + "data/amazon/book/reviews_Books_5.json"
# douban dataset original files
douban_movie = absolute_path + "data/douban/movie/moviereviews_cleaned.txt"
douban_music = absolute_path + "data/douban/music/musicreviews_cleaned.txt"

# CDR training test save folder
book_movie_save = absolute_path + "data/amazon/book-movie/"
movie_book_save = absolute_path + "data/amazon/movie-book/"
movie_music_save = absolute_path + "data/amazon/movie-music/"
music_movie_save = absolute_path + "data/amazon/music-movie/"
douban_movie_music_save = absolute_path + "data/douban/movie-music/"
douban_music_movie_save = absolute_path + "data/douban/music-movie/"

# preprocess user-mapping file path
user_map_paths = {
    "book": absolute_path + "data/amazon/book/user-map.txt",
    "music": absolute_path + "data/amazon/music/user-map.txt",
    "movie": absolute_path + "data/amazon/movie/user-map.txt",
    # "douban_movie": absolute_path + "data/douban/movie/user-map.txt",
    # "douban_music": absolute_path + "data/douban/music/user-map.txt",
}
item_map_paths = {
    "book": absolute_path + "data/amazon/book/item-map.txt",
    "music": absolute_path + "data/amazon/music/item-map.txt",
    "movie": absolute_path + "data/amazon/movie/item-map.txt",
    # "douban_movie": absolute_path + "data/douban/movie/item-map.txt",
    # "douban_music": absolute_path + "data/douban/music/item-map.txt",
}
overlap_save = absolute_path + "data/amazon/overlap/"
douban_overlap_save = absolute_path + "data/douban/overlap/"

train_data = {
    "book_movie_save": absolute_path + "data/amazon/book-movie/",
    "movie_music_save": absolute_path + "data/amazon/movie-music/",
    "movie_book_save": absolute_path + "data/amazon/movie-book/",
    "music_movie_save": absolute_path + "data/amazon/music-movie/",
    # "douban_movie_music_save": absolute_path + "data/douban/movie-music/",
    # "douban_music_movie_save": absolute_path + "data/douban/music-movie/",
}
main_weight = {
    "movie_music": absolute_path + "data/weight/main/movie-music.pt",
    "music_movie": absolute_path + "data/weight/main/music-movie.pt",
    "book_movie": absolute_path + "data/weight/main/book-movie.pt",
    "movie_book": absolute_path + "data/weight/main/movie-book.pt",
    # "douban_movie_music": absolute_path + "data/weight/main/douban_movie-music.pt",
    # "douban_music_movie": absolute_path + "data/weight/main/douban_music-movie.pt",
}
# pretrain weight save path
lightgcn_weight = {
    "book_100": absolute_path + "data/weight/lightgcn/book_100.pt",
    "movie_100": absolute_path + "data/weight/lightgcn/movie_100.pt",
    "music_100": absolute_path + "data/weight/lightgcn/music_100.pt",
    # "douban_movie_100": absolute_path + "data/weight/lightgcn/douban_movie_100.pt",
    # "douban_music_100": absolute_path + "data/weight/lightgcn/douban_music_100.pt",
}

# -1-1
RATES_MULTIPLY = 4
RATES_ADD = 1
EPS = 1e-10
# Define global hyperparameters
LIGHTGCN_LR = 0.001
EMBBEDDING_DIM = 32
NUM_LAYERS = 2  # Number of GCN layers for LightGCN
BATCH_SIZE = 2048 # 256
ITERATIONS = 30 # 10000
PRETRAIN_ITERATIONS = 1000
ITERS_PER_EVAL = 10  # 100
MAXITER = 50 # OT iter
LAMBDA_E = 0.1
WEIGHT_DECAY = 1e-5
OT_WEIGHT_DECAY = 1e-5
INDEX_LIST=[30, 953, 5275]
PRETRAIN_PATIENCE = 50  # Consecutive epochs without validation Recall@10 improvement.
MAPPING_PATIENCE = 10
ASHA_GRACE_PERIOD = 10
SEED = 2026
DOUBAN_PROCESSED_ROOT = Path(absolute_path) / "data/douban/processed"
DOUBAN_MIN_INTERACTIONS = 10
