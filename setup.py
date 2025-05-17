# SETUP PROJECT
from pathlib import Path
from typing import Literal
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from helpers.paths import create_directories
from helpers.cleaning import clean_and_save
from insights.maps.create_map import generate_map
from insights.maps.metro_distance_map import generate_metro_distance_map
from insights.price_distr import generate_price_distribution_summary
from insights.price_correlations import generate_price_correlation_analysis
from insights.metro_dist_price_corr import generate_price_metro_dist_corr_pic

def setup(dataset: Literal['13k','20k']):

    # 0 - CREATE ALL THE NECESSARY DIRECTORIES
    create_directories()
    # 1 - CLEANING AND SAVING CHOSEN DATASET 
    # Creating cluster-cell map
    clean_and_save(dataset=dataset,force=True)
    # 2 GENERATE ALL THE INSIGHTS MAPS
    generate_map()
    generate_metro_distance_map()

    # 3 GENERATE ALL THE STATISTICS PICTURES
    generate_price_distribution_summary()
    generate_price_correlation_analysis()
    generate_price_metro_dist_corr_pic()






if __name__ == "__main__":
    setup('13k')


