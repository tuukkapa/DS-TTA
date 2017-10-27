"""Module containing functions to create visualizations from data

WordCloud from https://github.com/amueller/word_cloud

In all of the word cloud functions kwargs are passed to the wordcloud object.
see: http://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
"""
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def word_cloud_from_file(inpath, outpath, **kwargs):
    """Creates word cloud from words in file."""
    with open(inpath, encoding='utf-8') as file:
        wc = WordCloud(**kwargs).generate(file.read())
        wc.to_file(outpath)


def word_cloud_from_string(string, outpath, **kwargs):
    """Creates word cloud from words in string."""
    wc = WordCloud(**kwargs).generate(string)
    wc.to_file(outpath)


def word_cloud_from_frequencies(freq, outpath, **kwargs):
    """Creates word cloud from word frequencies in dictionary."""
    wc = WordCloud(**kwargs).generate_from_frequencies(freq)
    wc.to_file(outpath)


def bar_plot_from_dataframe(df, outpath):
    """Creates bar plot for positivity

    Data frame expected to have index for tag names and 2 columns:
    One for positive and negative counts or percentages
    """
    df.div(df.sum(axis=1), axis=0).plot.barh(stacked=True, color=['tab:blue', 'tab:gray', 'tab:red'])
    #plt.show()
    plt.savefig(outpath)


if __name__ == '__main__':
    word_cloud_from_file(sys.argv[1], sys.argv[2])
