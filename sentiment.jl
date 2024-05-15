using VaderSentiment
using TextAnalysis
using SpacyTokenizers
using DataFrames

"""
Sentiment Scorer
"""
struct SentimentScorer
    analyzer::VaderSentimentAnalyzer
end

SentimentScorer() = SentimentScorer(VaderSentimentAnalyzer())

function score_sentiment(scorer::SentimentScorer, text::String)
    sentiment_scores = scorer.analyzer(text)
    return sentiment_scores.compound
end

"""
Sentiment Analyzer
"""
struct SentimentAnalyzer
    analyzer::VaderSentimentAnalyzer
    tokenizer::SpacyTokenizer
end

SentimentAnalyzer() = SentimentAnalyzer(VaderSentimentAnalyzer(), SpacyTokenizer("en_core_web_sm"))

function clean_text(text::String)
    clean_text = replace(text, "\n" => " ")
    clean_text = replace(clean_text, "/" => " ")
    clean_text = filter(c -> c != '\'', clean_text)
    return clean_text
end

function split_into_sentences(analyzer::SentimentAnalyzer, text::String)
    doc = analyze(analyzer.tokenizer, text)
    sentences = [sent.text for sent in doc.sents]
    return sentences
end

function get_vader_sentiment(analyzer::SentimentAnalyzer, text::String)
    sentiment_scores = analyzer.analyzer(text)
    return sentiment_scores.compound
end

function analyze_text(analyzer::SentimentAnalyzer, text_object)
    text = text_object.text
    cleaned_text = clean_text(text)
    sentences = split_into_sentences(analyzer, cleaned_text)

    vader_sentiments = Float64[]
    for sentence in sentences
        vader_score = get_vader_sentiment(analyzer, sentence)
        push!(vader_sentiments, vader_score)
    end

    sentiment_df = DataFrame(
        Sentence=sentences,
        Vader_Sentiment=vader_sentiments,
    )

    return sentiment_df
end