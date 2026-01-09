# Amazon Review Score Prediction

## What is This Project?

This project uses a computer program to automatically figure out if an Amazon product review has a low, medium, or high score. It looks at what people wrote in their reviews and other information about the reviews to make this prediction. We used more than 500,000 reviews to teach the program, and it can correctly predict the score level 84% of the time.

## Table of Contents
- [What Does This Do?](#what-does-this-do)
- [What Data Do We Use?](#what-data-do-we-use)
- [How to Set It Up](#how-to-set-it-up)
- [How to Use It](#how-to-use-it)
- [How Well Does It Work?](#how-well-does-it-work)
- [Want to Help?](#want-to-help)
- [License](#license)

## What Does This Do?

The main purpose of this project is to guess whether an Amazon review will have a low, medium, or high score. We clean up the review text and use special computer programs (called machine learning) to learn patterns from past reviews. Then the program can look at new reviews and predict what score level they should have. Our program got the right answer 84 out of 100 times when we tested it.

## What Data Do We Use?

We got our data from Kaggle, which is a website where people share datasets. The data comes in three files:

- **foods_training.csv**: This file has 518,358 reviews that we use to teach our program. Each review already has a score level (low, medium, or high) attached to it.
- **foods_testing.csv**: This file has 5,000 reviews that we use to test our program. These reviews don't have score levels, so our job is to predict what they should be.
- **sample_submission.csv**: This is a template file that shows how to format our predictions for submitting to Kaggle.

Each review in the data has these pieces of information:
- **ID** - A special number that identifies each review
- **productID** - A special number that identifies which product the review is about
- **userId** - A special number that identifies who wrote the review
- **helpfulness** - A rating showing how helpful other people found the review
- **summary** - A short summary of what the review says
- **text** - The full review text that the customer wrote
- **score_level** - The answer we're trying to predict (low, medium, or high)

## How to Set It Up

First, you need to install the computer programs (called libraries) that this project needs. Open your terminal or command prompt and type:

```bash
pip install -r requirements.txt
```

This will download and install all the necessary tools.

## How to Use It

Follow these steps:

1. **Get the data files:**
   - Go to Kaggle and download the dataset
   - Put all the CSV files into a folder called `data` in your project

2. **Train the program:**
   - Open your terminal or command prompt
   - Type this command:
   ```bash
   python train_model.py
   ```
   - This will teach the program how to predict score levels. It might take a few minutes.

3. **Make predictions:**
   - After training is done, use this command to predict scores for the test reviews:
   ```bash
   python predict.py --input data/foods_testing.csv --output submission.csv
   ```
   - This creates a file called `submission.csv` with all the predictions.

4. **Submit your results:**
   - Take the `submission.csv` file and upload it to Kaggle to see how well your predictions did.

## How Well Does It Work?

When we tested our program on new reviews it had never seen before, it correctly predicted the score level 84% of the time. This means out of 100 reviews, it got 84 right.

## Want to Help?

If you have ideas to make this project better, we'd love your help! Feel free to share your suggestions or improvements.

## License

Anyone can use this project for free. It's available under the MIT License.
