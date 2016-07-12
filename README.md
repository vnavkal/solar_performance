# kWh Analytics Data Science Interview Project

Welcome to the kWh Analytics data science interview project. Our goal in this project is to both (a) show you an example of the problems we solve, and (b) get an idea of how you approach the problem. We hope you walk away understanding a bit more of what would be expected of you. The data provided in this project is fictional, but the problem we ask you to solve is relevant.

## Context

There is a wide variety of solar panels on the market. Photovoltaic panel models vary in efficiency and durability based on the materials and manufacturing practices. Given field performance data, can we determine if a panel manufacturer's quality is changing with time?

## Question

In the file "data/datasci_takehome.csv.gz", located in this repository, we provide monthly performance data from systems that all use panels from manufacturer X. Each system also has a particular vintage (A, B, or C) corresponding to the year that the panel was manufactured.

The performance comes in the form of expectedkwh (the expected energy produced in kWh) and correctedkwh (the actual energy produced, weather corrected, in kWh). The metric that we use to benchmark performance is called the Performance Ratio (PR). In this case, the PR is computed as correctedkwh / expectedkwh. In any given month, a system that under-performs its expectation has a PR less than 1, and a system that over-performs its expectation has a PR greater than 1.

The question to answer: Is there reason to believe that panel manufacturer X's quality is changing year-to-year?

## Instructions

1. Clone the repository or just download the data file.
2. Submit your work through email as a zipped repository or directory.

## General Guidelines

- Version control is not required.
- Please, do not submit a pull request.
- Explain your answer. We want to follow your thought process.
- Support your answer with valid code.
- Feel free to use tables or visualizations if they help support your answer.
- Use the tools you are most comfortable with. You can submit your answer and code in any format.
- Work on the problem at your convenience.
- Please, take no longer than 4 hours.
