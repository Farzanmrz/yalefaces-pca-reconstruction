# Face Reconstruction using PCA

This project demonstrates how to reconstruct faces from the Yale Faces dataset using Principal Component Analysis (PCA). PCA is a statistical technique used to emphasize variation and capture strong patterns in a dataset. This project uses PCA to reconstruct faces by projecting the data onto principal components and creating a video to show the reconstruction process for a randomly selected image.

## Project Structure

- `reconstruct.ipynb`: Jupyter Notebook containing the step-by-step implementation.
- `reconstruct.py`: Python script generated from the notebook.
- `yalefaces/`: Directory containing the dataset images.



## Running the Project

### Setup

To set up the environment, ensure you have the required libraries. You can install them using pip:

```bash
pip install numpy matplotlib pillow opencv-python
```

### Using Jupyter Notebook

1. Open the `reconstruct.ipynb` file in Jupyter Notebook.
2. Run all cells to execute the code step-by-step.

### Using the Python Script

1. Ensure you have the `yalefaces` directory in the same folder as the script.
2. Run the script using the following command:

```bash
python reconstruct.py
```

## Expected Output

The project will generate a video file named `reconstruction.avi`, which demonstrates the step-by-step reconstruction of a randomly selected face from the dataset using PCA.

## Future Work

- Implement more sophisticated reconstruction techniques.
- Enhance the project with additional datasets for a more comprehensive analysis.
- Include more examples and tutorials on PCA and its applications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact 
Farzan Mirza: [farzanmrz@gmail.com](mailto:farzanmrz@gmail.com) | [farzan.mirza@drexel.edu](mailto:farzan.mirza@drexel.edu) | [LinkedIn](https://www.linkedin.com/in/farzan-mirza13/)
