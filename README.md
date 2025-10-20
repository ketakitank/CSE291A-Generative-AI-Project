# LLM for Stock Price Direction Prediction 📈

## Introduction

Trying to predict the stock market is a notoriously hard problem. The standard thinking, known as the **Efficient Market Hypothesis**, basically says it's impossible because prices already reflect all known information. But we know that public sentiment and market weirdness do have an impact. That's where our project comes in. We want to see if we can get better predictions by looking at both the numbers (price history) and the words (news and social media).

Our idea is to use a Large Language Model (LLM) and fine-tune it on a special **unified textual representation** that combines historical price data with news articles, all in an end-to-end system.

---

### Problem Definition

Our main goal is pretty straightforward: we're trying to predict whether a stock's price will **go up or down tomorrow**. We're setting this up as a binary classification problem. The input for our model at any given time $t$ is a combination of two things: the recent price history ($P_t$) and any relevant news articles from that day ($N_t$). Our model, a fine-tuned LLM we're calling $\mathcal{L}$, will then learn a function to map that input to a simple 'UP' or 'DOWN' prediction.

---

### Problem Significance

LLMs are usually known for creating text, so seeing if they can handle a tricky classification task with two different kinds of data is a great test. We're really pushing the model to understand the subtle connections between a news story and what the market does next. This project is a strong case for end-to-end learning. Instead of the old way of crunching news down to a single "sentiment score" (and losing a ton of info), we feed everything in raw. If this works, it could lead to better financial prediction tools, but the same idea could be used for other things too, like spotting supply chain issues from reports or tracking economic trends. 🤖

---

### Technical Challenges

We're expecting to tackle a few key challenges during this project:

1.  **Getting the data right:** As papers like [webscraping] point out, just getting clean, time-stamped financial news and aligning it perfectly with price data is a huge job. Also, as [karlemstr2021using] mentions, social media is full of noise, so filtering that out will be tough.
2.  **Designing the model:** Our biggest hurdle is figuring out how to represent numerical price history in a way that a text-based LLM can actually understand. If we just turn numbers into a string, the model might miss important patterns.
3.  **Implementation:** Fine-tuning a big model like Llama 3 requires access to powerful GPUs. Getting the PEFT/LoRA setup working right will be a real engineering task.
4.  **Evaluation:** The market is always changing. A model that works great on last year's data might completely fail next year. We have to be really careful to build a back-testing system that's realistic and doesn't accidentally "peek" into the future.

---

### State-of-the-Art

When we looked at other papers, we saw a common pattern. Most of them handle numbers and text in two separate steps. For example, papers like [stockpredictiontwitter] and [karlemstr2021using] will first run news or tweets through a sentiment analyzer (like VADER) to get a simple score. Then, they feed that score into a totally separate model, like an LSTM, along with the price data.

The main problem with this approach is that you lose a lot of information. A whole news article gets boiled down to just "positive" or "negative," making it impossible for the model to catch more complex relationships. Even newer models like FinBERT [araci2019finbert], which are better at understanding financial text, are still just used as a pre-processing step. We're doing something different by feeding everything—the raw text and the price data—directly into one big model and letting it figure out the connections on its own.

---

### Our Contributions

Here’s a quick summary of what's new about our project:

-   Our main contribution is a new method that combines price history and news into a **single text-based prompt** that an LLM can process in one go.
-   We’re using the built-in knowledge of a large pre-trained model to understand financial news in a much deeper way than just sentiment analysis.
-   We're showing how to use efficient methods like **PEFT and LoRA** to make this kind of fine-tuning possible without needing a supercomputer.
