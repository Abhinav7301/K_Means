# K-Means Clustering

A simple and efficient implementation of the K-Means clustering algorithm with interactive visualization.

## Overview

K-Means is an unsupervised machine learning algorithm that partitions data into k distinct clusters by minimizing the variance within each cluster. This implementation provides an easy-to-use interface for clustering analysis and visualization.

## Features

- **Simple Implementation**: Clean and easy-to-understand code
- **Interactive Visualization**: Visual representation of clusters and centroids
- **Customizable Parameters**: Adjust number of clusters and iterations
- **Real-world Examples**: Pre-loaded datasets for quick testing
- **Model Persistence**: Save and load trained models

## Project Structure

```
K_Means/
├── app.py                      # Main application file
├── k_means.ipynb              # Jupyter notebook with examples
├── kmean+.ipynb               # Advanced clustering examples
├── k_means_model.pkl          # Trained model
├── Mall_Customers (3).csv     # Sample dataset
├── mall_customers_clustered.csv # Clustering results
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abhinav7301/K_Means.git
cd K_Means
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Required packages:
   - numpy
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn

## Usage

### Running the Application

```bash
python app.py
```

### Using Jupyter Notebooks

```bash
jupyter notebook k_means.ipynb
```

## How K-Means Works

1. **Initialization**: Randomly select k initial centroids
2. **Assignment**: Assign each point to the nearest centroid
3. **Update**: Recalculate centroids based on cluster assignments
4. **Convergence**: Repeat steps 2-3 until centroids no longer change

## Parameters

- `n_clusters`: Number of clusters to form (k)
- `max_iter`: Maximum number of iterations (default: 300)
- `random_state`: Random seed for reproducibility
- `init`: Initialization method (k-means++, random)

## Example Output

The application generates:
- Scatter plots showing clusters and centroids
- Cluster assignments for each data point
- Within-cluster sum of squares (WCSS) plot
- Cluster statistics and summaries

## Algorithm Complexity

- **Time Complexity**: O(n * k * i * d)
  - n: number of data points
  - k: number of clusters
  - i: number of iterations
  - d: number of dimensions

- **Space Complexity**: O(n * d)

## Applications

- Customer segmentation
- Image compression
- Document clustering
- Anomaly detection
- Data preprocessing

## Best Practices

- Standardize/normalize your data before clustering
- Use the elbow method to determine optimal k
- Try multiple initializations for better results
- Evaluate clusters using silhouette analysis

## Known Limitations

- Assumes spherical clusters
- Sensitive to initial centroid placement
- May converge to local optimum
- Not suitable for very large datasets without optimization

## Future Improvements

- [ ] Mini-batch K-means for large datasets
- [ ] Multiple initialization strategies
- [ ] Automated k selection (elbow method)
- [ ] GPU acceleration
- [ ] Additional distance metrics

## Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest enhancements
- Submit pull requests

## License

MIT License - Open source and free to use

## References

- [K-Means on Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
- [Scikit-learn K-Means Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [K-Means Tutorial](https://www.geeksforgeeks.org/k-means-clustering/)

---

**Last Updated**: February 2026
**Author**: Abhinav7301
