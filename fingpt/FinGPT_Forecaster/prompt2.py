import os
import json
import random
import finnhub
import yfinance as yf
import pandas as pd
from openai import OpenAI

from indices import *

finnhub_client = finnhub.Client(api_key=os.environ.get("FINNHUB_KEY"))



# Prompt structure
# Instruction: ...
# [Company Introduction]
# [Returns]
# - From week A to week B: ...
# [Index]
# [Relevant News]
# Final Instruction


def get_company_prompt(symbol):
    
    profile = finnhub_client.company_profile2(symbol=symbol)

    company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
        "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."

    formatted_str = company_template.format(**profile)
    
    return formatted_str


def get_crypto_prompt(symbol):

    profile = yf.Ticker(symbol).info

    crpyto_template = """[Cryptocurrency Introduction]: {description}. It has a market capilization of {marketCap}."""
    
    formatted_str = crpyto_template.format(**profile)
    
    return formatted_str


def get_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    # head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. News during this period are listed below:\n\n".format(
    #     start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    head = f"From {start_date} to {end_date}, news during this period are listed below:\n\n"
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    basics = json.loads(row['Basics'])
    if basics:
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."
    
    return head, news, basics


def get_crypto_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    head = "From {} to {}, {}'s price {} from {:.2f} to {:.2f}. News during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    return head, news, None

def get_returns_by_row(symbol, row):
    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    time_series = f"From {start_date} to {end_date}, {symbol}'s stock price {term} from {row['Start Price']:.2f} to {row['End Price']:.2f} = {100*row['Weekly Returns']:.2f}% => {row['Bin Label']}"
    return time_series


def get_index_returns(symbol, row):
    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    time_series = f"From {start_date} to {end_date}, {symbol}'s stock price {term} from {row['Index Start Price']:.2f} to {row['Index End Price']:.2f} = {100*row['Index Weekly Returns']:.2f}%"
    return time_series


def sample_news(news, k=5):
    
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


# def map_bin_label(bin_lb):
    
#     lb = bin_lb.replace('U', 'up by ')
#     lb = lb.replace('D', 'down by ')
#     lb = lb.replace('1', '0-1%')
#     lb = lb.replace('2', '1-2%')
#     lb = lb.replace('3', '2-3%')
#     lb = lb.replace('4', '3-4%')
#     if lb.endswith('+'):
#         lb = lb.replace('5+', 'more than 5%')
# #         lb = lb.replace('5+', '5+%')
#     else:
#         lb = lb.replace('5', '4-5%')
    
#     return lb


def map_bin_label(bin_lb):
    # Check if the label indicates "up by" or "down by"
    direction = "up by" if bin_lb.startswith('U') else "down by"
    
    # Extract the number part of the label
    number_part = bin_lb[1:]
    
    # Handle the '100+' case explicitly
    if number_part == "100+":
        return f"{direction} more than 100%"
    
    try:
        # Convert the number part to an integer
        bin_num = int(number_part)
    except ValueError:
        # Return an error message for invalid inputs
        return "Invalid bin label"
    
    # For numbers up to 99, calculate the percentage range
    if 0 < bin_num <= 100:
        lb = f"{direction} {bin_num-1}-{bin_num}%"
    else:
        lb = "Invalid bin label"
    
    return lb



PROMPT_END = {
    "company": "\n\nBased on all the information before {start_date}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
        "Then let's assume your prediction for next week ({start_date} to {end_date}) is {prediction}. Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis.",

    "crypto": "\n\nBased on all the information before {start_date}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from cryptocurrencies related news. " \
        "Then let's assume your prediction for next week ({start_date} to {end_date}) is {prediction}. Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis."
}

def get_all_prompts(symbol, index_name, data_dir, start_date, end_date, min_past_weeks=1, max_past_weeks=3, n_articles=3, with_basics=True):

    
    if with_basics:
        df = pd.read_csv(f'{data_dir}/{symbol}_{start_date}_{end_date}.csv')
    else:
        df = pd.read_csv(f'{data_dir}/{symbol}_{start_date}_{end_date}_nobasics.csv')
    
    if symbol in CRYPTO:
        info_prompt = get_crypto_prompt(symbol)
    else:
        info_prompt = get_company_prompt(symbol)

    prev_rows = []
    all_prompts = []
    all_rows = []
    for row_idx, row in df.iterrows():
        prompt = "\n[Related News]\n"
        returns = "\n\n[Returns]\nThe stock price movement has been as follows:\n\n"
        if symbol in CRYPTO:
            index_returns = ""
        else:
            index_returns = f"\n\n[Index Returns]\n{symbol} stock is contained in the index {index_name}. The behavior of this index has been the following:\n\n"
        rows = []
        if len(prev_rows) >= min_past_weeks:
            idx = min(random.choice(range(min_past_weeks, max_past_weeks+1)), len(prev_rows))
            for i in range(-idx, 0):
                # Add Price Movement (Head)
                prompt += "\n" + prev_rows[i][0]
                # Add News of previous weeks
                sampled_news = sample_news(
                    prev_rows[i][1],
                    min(n_articles, len(prev_rows[i][1]))
                )
                if sampled_news:
                    prompt += "\n".join(sampled_news)
                else:
                    prompt += "No relative news reported."

                returns += prev_rows[i][3] + '\n'
                if symbol not in CRYPTO:
                    index_returns += prev_rows[i][4] + '\n'

                rows.append(prev_rows[i][5])

        if symbol in CRYPTO:
            head, news, basics = get_crypto_prompt_by_row(symbol, row)
            index_time_series = None
        else:
            head, news, basics = get_prompt_by_row(symbol, row)
            index_time_series = get_index_returns(index_name, row)

        time_series = get_returns_by_row(symbol, row)


        prev_rows.append((head, news, basics, time_series, index_time_series, row))
        if len(prev_rows) > max_past_weeks:
            prev_rows.pop(0)  

        if not prompt:
            continue

        prediction = map_bin_label(row['Bin Label'])

        if with_basics:
            prompt = info_prompt + '\n'  + prompt + '\n' + index_returns +'\n' + basics + '\n' + returns
        else:
            prompt = info_prompt + '\n' + prompt + '\n' + index_returns + "\n" + returns

        prompt += PROMPT_END['crypto' if symbol in CRYPTO else 'company'].format(
            start_date=row['Start Date'],
            end_date=row['End Date'],
            prediction=prediction,
            symbol=symbol
        )

        if len(rows) == 0:
            continue

        all_prompts.append(prompt.strip())
        rows.append(row)
        all_rows.append(rows)

    return all_prompts, all_rows

