{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# COSC 89.21 | Professor Vosoughi | Winter 2021\n",
    "\n",
    "# Final Project: Predicting StackOverflow R Question Binary Rating (Positive or Negative Score) from Question Text Data\n",
    "\n",
    "# By Clayton Bass (Undergraduate '22)\n",
    "\n",
    "## The emphasis of this project is on exploring and tackling the cleaning/pre-processing process for a very messy, unstructured dataset that includes lots of code, HTML tags, formal mathematical/programming language, and not much sentiment. However, modeling is still included so as to answer the question at hand."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Section 0: Loading the Data\n",
    "### Data Source (Kaggle): https://www.kaggle.com/stackoverflow/rquestions?select=Questions.csv (Files Downloaded: 'Answers.csv', 'Questions.csv', 'Tags.csv'; File Used: 'Questions.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import a few relevant packages to begin\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import nltk\n",
    "import re"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import the data \n",
    "questions_raw = pd.read_csv('Questions.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Change the dates column to pandas datetime object\n",
    "questions_raw['CreationDate'] = pd.to_datetime(questions_raw['CreationDate'])\n",
    "questions_raw.index = questions_raw['CreationDate'] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Look only at the questions from 2016 and 2017, so that\n",
    "# we examine only the more recent R versions (since the code\n",
    "# is harder to interpret from older versions);\n",
    "# however, get rid of all questions that have \"0\" as their score,\n",
    "# since there is no insight (we will assume) into their quality,\n",
    "# acknowledging that some questions may be controversial\n",
    "# and get many upvotes and many downvotes thathappen to equalize\n",
    "# at 0. \n",
    "questions_body_2016 = questions_raw['Body']['2016']\n",
    "questions_body_2017 = questions_raw['Body']['2017']\n",
    "questions_body = pd.concat([questions_body_2016, questions_body_2017],\n",
    "                          axis=0)\n",
    "\n",
    "questions_title_2016 = questions_raw['Title']['2016']\n",
    "questions_title_2017 = questions_raw['Title']['2017']\n",
    "questions_title = pd.concat([questions_title_2016, questions_title_2017],\n",
    "                          axis=0)\n",
    "\n",
    "score_2016 = questions_raw['Score']['2016']\n",
    "score_2017 = questions_raw['Score']['2017']\n",
    "score = pd.concat([score_2016, score_2017],\n",
    "                          axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Put together all of the questions' titles and bodies from the\n",
    "# proper year\n",
    "questions_full = pd.concat([score, questions_title, questions_body],\n",
    "                          axis = 1)\n",
    "questions_full['CreationDate'] = questions_full.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove all rows where there is a score of 0, for the reasons mentioned\n",
    "# a couple of cells above (there is no sentimental value to such questions)\n",
    "questions_full = questions_full[questions_full['Score'] != 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(40736, 4)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Notice that we have 40736 samples, which should be enough for our purposes\n",
    "questions_full.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Turn 'Score' into binary: 1 if strictly positive, 0 if strictly negative\n",
    "# (since all scores of 0 have been filtered out)\n",
    "questions_full['score_binary'] = questions_full['Score'].apply(lambda x: 1 if x > 0 else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Score</th>\n",
       "      <th>Title</th>\n",
       "      <th>Body</th>\n",
       "      <th>CreationDate</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>score_binary</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5175</td>\n",
       "      <td>5175</td>\n",
       "      <td>5175</td>\n",
       "      <td>5175</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>35561</td>\n",
       "      <td>35561</td>\n",
       "      <td>35561</td>\n",
       "      <td>35561</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Score  Title   Body  CreationDate\n",
       "score_binary                                   \n",
       "0              5175   5175   5175          5175\n",
       "1             35561  35561  35561         35561"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Notice that the data are quite unbalanced, in a ratio of about 1:7\n",
    "# for class 0:class 1. We therefore will downsample the positive class by half.\n",
    "questions_full.groupby(['score_binary']).count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<p>What's the best way to get the label of an <strong>actionButton</strong> that is clicked? I have an <strong>actionButton</strong> that has the label updated. When the user clicks, I need to capture that. I tried  input$action2.label and input$action2.text, but neither worked.</p>\n",
      "\n",
      "<p><strong>ui.R</strong></p>\n",
      "\n",
      "<pre><code>library(shiny)\n",
      "\n",
      "shinyUI( fluidPage(\n",
      "\n",
      "tags$head(tags$style(\n",
      "  HTML('\n",
      "       { background-color: #dec4de;}\n",
      "       #action4 { width: 275 px; color:red; background-color:yellow }\n",
      "       #action1, #action2, #action3 { color:black; background-color:lime }\n",
      "       body, label, input, button, select { font-family: \"Arial\"; }'\n",
      "  )\n",
      ")),    \n",
      "\n",
      "titlePanel(\"My Shiny App!\"),\n",
      "\n",
      "sidebarLayout(\n",
      "  sidebarPanel(\n",
      "    tabsetPanel(type = \"tabs\", \n",
      "                tabPanel(\"About\", \"Developer Info here\"), \n",
      "                tabPanel(\"How to Use\", \"User Docs\"))\n",
      "\n",
      "  ),\n",
      "\n",
      "  mainPanel(\n",
      "    img(src=\"capstone1.jpg\", align = \"center\"),\n",
      "    br(),br(),\n",
      "    tags$textarea(id=\"stext\", rows=3, cols=80, \"Default value\"),\n",
      "    br(),br(),br(),\n",
      "    actionButton(\"action1\", \"Action 1\"),\n",
      "    actionButton(\"action2\", \"Action 2\"),\n",
      "    actionButton(\"action3\", \"Action 3\"),\n",
      "    br(), br(),\n",
      "    actionButton(\"action4\", \n",
      "                   label = \"High &lt; &lt; &lt; &lt;  PROBABILITY  &gt; &gt; &gt; &gt; Low\")\n",
      "  )\n",
      ")))\n",
      "</code></pre>\n",
      "\n",
      "<p><strong>server.R</strong></p>\n",
      "\n",
      "<pre><code>library(shiny)\n",
      "\n",
      "shinyServer( function(input, output, session) {\n",
      "\n",
      "   observeEvent(input$action1, {\n",
      "      x &lt;- input$stext\n",
      "      print(input$action2.text)\n",
      "      updateTextInput(session, \"stext\", value=paste(x, \"Btn 1\"))\n",
      "   })\n",
      "\n",
      "   observeEvent(input$action2, {\n",
      "      x &lt;- input$stext\n",
      "      print(input$action2.label)\n",
      "      updateTextInput(session, \"stext\", value=paste(x, \"Btn 2\"))\n",
      "   })  \n",
      "\n",
      "   observeEvent(input$action3, {\n",
      "      x &lt;- input$stext\n",
      "      print(x)\n",
      "      updateTextInput(session, \"stext\", value=paste(x, \"Btn 3\"))\n",
      "   })  \n",
      "\n",
      "})\n",
      "</code></pre>\n",
      "\n",
      "<p>UPDATE: Adding code for shinyBS</p>\n",
      "\n",
      "<p><strong>ui.R</strong></p>\n",
      "\n",
      "<pre><code>{\n",
      "   library(shinyBS)\n",
      "   ....\n",
      "   bsButton(\"myBtn\", \"\")\n",
      "   ...\n",
      "\n",
      "}\n",
      "</code></pre>\n",
      "\n",
      "<p><strong>server.R</strong></p>\n",
      "\n",
      "<pre><code>{\n",
      "     ....\n",
      "     updateButton(session, \"myBtn\", \"New Label\")\n",
      "     ....\n",
      "}\n",
      "</code></pre>\n",
      "\n",
      "----------------------------------------\n",
      "<p>I seem to be having trouble setting up a ribbon in ggplot2 to display. </p>\n",
      "\n",
      "<p>Here's a made up data set:</p>\n",
      "\n",
      "<pre><code>&gt; GlobalDFData\n",
      "  Estimate Upper  Lower  Date   Area\n",
      "1      100   125    75 Q1_16 Global\n",
      "2      125   150   100 Q2_16 Global\n",
      "3      150   175   125 Q3_16 Global\n",
      "4      175   200   150 Q4_16 Global\n",
      "</code></pre>\n",
      "\n",
      "<p>Here's the code that I'm trying with no success. I get the line chart but not the upper and lower bounds</p>\n",
      "\n",
      "<pre><code>ggplot(GlobalDFData, aes(x = Date)) + \n",
      "  geom_line(aes(y = Estimate, group = Area, color = Area))+\n",
      "  geom_point(aes(y = Estimate, x = Date))+\n",
      "  geom_ribbon(aes(ymin = Lower, ymax = Upper))\n",
      "</code></pre>\n",
      "\n",
      "----------------------------------------\n",
      "<p>I have a data frame like this:</p>\n",
      "\n",
      "<pre><code>  message.id sender recipient\n",
      "1          1      A         B\n",
      "2          1      A         C\n",
      "3          2      A         B\n",
      "4          3      B         C\n",
      "5          3      B         D\n",
      "6          3      B         Q\n",
      "</code></pre>\n",
      "\n",
      "<p>I would like to summarize it by the counts of values in the sender and recipient columns to get this:</p>\n",
      "\n",
      "<pre><code>  address messages.sent messages.received\n",
      "1       A             3                 0\n",
      "2       B             3                 2\n",
      "3       C             0                 2\n",
      "4       D             0                 1\n",
      "5       Q             0                 1\n",
      "</code></pre>\n",
      "\n",
      "<p>I have working code, but it's messy, and I'm hoping there's a way to do this all in one <code>magrittr</code> chain instead of what I have below:</p>\n",
      "\n",
      "<pre><code>df &lt;- data.frame(message.id = c(1,1,2,3,3,3),\n",
      "                 sender = c(\"A\",\"A\",\"A\",\"B\",\"B\",\"B\"),\n",
      "                 recipient = c(\"B\",\"C\",\"B\",\"C\",\"D\",\"Q\"))\n",
      "sent &lt;- df %&gt;% \n",
      "  group_by(sender) %&gt;%\n",
      "  summarise(messages.sent = n()) %&gt;%\n",
      "  mutate(address = sender) %&gt;%\n",
      "  select(address, messages.sent)\n",
      "\n",
      "received &lt;- df %&gt;% \n",
      "  group_by(recipient) %&gt;%\n",
      "  summarise(messages.received = n()) %&gt;%\n",
      "  mutate(address = recipient) %&gt;%\n",
      "  select(address, messages.received)\n",
      "\n",
      "df_summary &lt;- merge(sent, received, all = TRUE) %&gt;%\n",
      "  replace(is.na(.), 0)\n",
      "</code></pre>\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Let's look at a few of the questions to see the kind\n",
    "# of cleaning we'll need to do. We will need to extract\n",
    "# the English text data, so we will remove all of the code\n",
    "# tags (and HTML tags) and the code in between tags so that\n",
    "# we are left with English text. We will ultimately engage\n",
    "# in 3 distinct levels/depths of text cleaning.\n",
    "print(questions_full.iloc[0,2])\n",
    "print('----------------------------------------')\n",
    "print(questions_full.iloc[1,2])\n",
    "print('----------------------------------------')\n",
    "print(questions_full.iloc[2,2])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Section I: Removing Code Tags/Initial Cleaning; Downsampling; Splitting into Train and Test Sets"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### I.i Remove Code Tags"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Write a function to remove all code tags in our dataset.\n",
    "# This will serve as the \"base\" dataset, on top of which we \n",
    "# will run all further cleaning and experimentation (since\n",
    "# the code and tags are not going to be considered really as\n",
    "# content in our model for the purposes of this project)\n",
    "def remove_tags(body): \n",
    "    # Lowercase the body\n",
    "    body = body.lower()\n",
    "    \n",
    "    # Tokenize the body\n",
    "    body_tokenized = nltk.word_tokenize(body)\n",
    "\n",
    "    # Turn back into a string (because some spacing issues between\n",
    "    # lines, where regular expressions start to fail; this isn't\n",
    "    # really cleaning the data at all)\n",
    "    body = ''\n",
    "    for word in body_tokenized:\n",
    "        body += word + ' ' \n",
    "       \n",
    "    # Remove everything within HTML tags, including those tags\n",
    "    # (after scrolling through hundreds of questions to see the\n",
    "    # most common HTML tags!)\n",
    "    body = re.sub(r'< code >.+< /code >', ' ', body)\n",
    "    body = re.sub(r'< pre > < code >[.|\\s]{3,}< /code > < /pre >', ' ', body)\n",
    "    body = re.sub(r'< blockquote >.+< /blockquote >', ' ', body)\n",
    "    body = re.sub(r'< href=.+>.+< /a >',' ', body)\n",
    "    body = re.sub(r'< li >.+< /li >', ' ', body)\n",
    "    body = re.sub(r'< ol >.+< /ol >', ' ', body)\n",
    "    \n",
    "    # Get rid of remaining tags that may not have been captured by the above\n",
    "    body = re.sub('< p >', ' ', body)\n",
    "    body = re.sub('< /p >', ' ', body)\n",
    "    body = re.sub('< /pre >', ' ', body)\n",
    "    body = re.sub('< pre >', ' ', body)\n",
    "    \n",
    "    return body"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Apply our function to the data\n",
    "questions_full['Body'] = questions_full['Body'].apply(lambda x: remove_tags(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### I.ii. Downsampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Downsample the data, then split into train and test sets\n",
    "# (downsample by about half in the positive class)\n",
    "\n",
    "# Create a dataframe with only the positive class by masking\n",
    "questions_positive = questions_full[questions_full['score_binary'] == 1]\n",
    "questions_positive_downsampled = questions_positive.sample(frac = 0.5,\n",
    "                                                          random_state=0)\n",
    "\n",
    "# Create a dataframe with only the negative class by masking\n",
    "questions_negative = questions_full[questions_full['score_binary'] == 0]\n",
    "\n",
    "# Re-connect the two dataframes back into one\n",
    "questions_downsampled = pd.concat([questions_positive_downsampled, questions_negative],\n",
    "                                 axis=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### I.iii. Split into Train and Test Sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(questions_downsampled.drop(['score_binary','Score'],axis=1),\n",
    "                                                   questions_downsampled['score_binary'], random_state=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Section II: Two Variations of Data Cleaning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### II.a. COMPLETE Data Cleaning\n",
    "\n",
    "I will now clean the data for the purpose of extracting TFIDF and CountVectorizer features from the data in the next section. I use a differently cleaned dataset for such features because I may want to use some of the messier aspects of the original data (such as symbols or code objects) as features, and I may remove these elements in the cleaning process for TFIDF/CountVectorizer specifically, since TFIDF/CountVectorizer would not work well with punctuation and other LaTeX/HTML commands inside.\n",
    "\n",
    "There will be one more subsection here where we clean the data slightly less, just to see how that might impact the final results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "IIa_df = X_train.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean the 'Title' column\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "def fully_clean_title(title):\n",
    "    # Lowercase the title\n",
    "    title = title.lower()\n",
    "    \n",
    "    # Tokenize the title and remove all stopwords\n",
    "    title_tokenized = nltk.word_tokenize(title)\n",
    "    my_stopwords = set(stopwords.words('english'))\n",
    "    title_tokenized = [word for word in title_tokenized if word not in my_stopwords]\n",
    "    \n",
    "    # Remove all punctuation\n",
    "    set_punctuation = {'?','.',',','!','@','#',\n",
    "                       '%','^','&','*','(',')','-',\n",
    "                       '{','}','[',']','\\\\',\"|\",':',\n",
    "                       ';','<','>','/','$','~','`','\"\"','\\'',\n",
    "                      '``',\"''\", \"—\",\"'\",\"’\",\"'s\"}\n",
    "    WNL = nltk.WordNetLemmatizer()\n",
    "\n",
    "    stemmed_words = [WNL.lemmatize(word) for word in title_tokenized\n",
    "              if word not in set_punctuation]\n",
    "    \n",
    "    # Turn back into a string\n",
    "    mystring = ''\n",
    "    for word in stemmed_words:\n",
    "        mystring += word + ' '\n",
    "        \n",
    "    return mystring"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean the 'Body' column\n",
    "import re\n",
    "\n",
    "def fully_clean_body(body):\n",
    "    # Tokenize the body and remove all stopwords\n",
    "    body_tokenized = nltk.word_tokenize(body)\n",
    "    my_stopwords = set(stopwords.words('english'))\n",
    "    body_tokenized = [word for word in body_tokenized if word not in my_stopwords]\n",
    "\n",
    "    # Remove all punctuation\n",
    "    set_punctuation = {'?','.',',','!','@','#',\n",
    "                       '%','^','&','*','(',')','-',\n",
    "                       '{','}','[',']','\\\\',\"|\",':',\n",
    "                       ';','<','>','/','$','~','`','\"\"','\\'',\n",
    "                      '``',\"''\", \"—\",\"'\",\"’\",\"'s\"}\n",
    "    \n",
    "    # Lemmatize the words (since this is more accurate\n",
    "    # then using the PorterStemmer, generally speaking)\n",
    "    WNL = nltk.WordNetLemmatizer()\n",
    "\n",
    "    stemmed_words = [WNL.lemmatize(word) for word in body_tokenized\n",
    "              if word not in set_punctuation]\n",
    "    \n",
    "    # Turn back into a string\n",
    "    mystring = ''\n",
    "    for word in stemmed_words:\n",
    "        mystring += word + ' '\n",
    "    \n",
    "    return mystring"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean the 'Title' column into a new column\n",
    "IIa_df['title_cleaned'] = IIa_df['Title'].apply(lambda x: fully_clean_title(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean the 'Body' column into a new column\n",
    "IIa_df['body_cleaned'] = IIa_df['Body'].apply(lambda x: fully_clean_body(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Concatenate the 'Body' and 'Title' text to make modeling easier\n",
    "IIa_df['text_cleaned'] = IIa_df['title_cleaned'] + ' ' + IIa_df['body_cleaned']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Section II.b.: LESS COMPLETE Data Cleaning\n",
    "\n",
    "Now, we clean the data a little less completely and see (ultimately in section III) how this compares with the fully cleaned dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "IIb_df = X_train.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean the 'Title' column only partially, without\n",
    "# lemmatization\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "def partially_clean_title(title):\n",
    "    # Lowercase the title\n",
    "    title = title.lower()\n",
    "    \n",
    "    # Tokenize the title and remove all stopwords\n",
    "    title_tokenized = nltk.word_tokenize(title)\n",
    "    my_stopwords = set(stopwords.words('english'))\n",
    "    title_tokenized = [word for word in title_tokenized if word not in my_stopwords]\n",
    "    \n",
    "    # Remove all punctuation\n",
    "    set_punctuation = {'?','.',',','!','@','#',\n",
    "                       '%','^','&','*','(',')','-',\n",
    "                       '{','}','[',']','\\\\',\"|\",':',\n",
    "                       ';','<','>','/','$','~','`','\"\"','\\'',\n",
    "                      '``',\"''\", \"—\",\"'\",\"’\",\"'s\"}\n",
    "    \n",
    "    title_tokenized = [word for word in title_tokenized if\n",
    "                      word not in set_punctuation]\n",
    "    \n",
    "    # Turn back into a string\n",
    "    mystring = ''\n",
    "    for word in title_tokenized:\n",
    "        mystring += word + ' '\n",
    "        \n",
    "    return mystring"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Only partially clean the 'Body' column, without lemmatization as before\n",
    "import re\n",
    "\n",
    "def partially_clean_body(body):\n",
    "    # Tokenize the body and remove all stopwords\n",
    "    body_tokenized = nltk.word_tokenize(body)\n",
    "    my_stopwords = set(stopwords.words('english'))\n",
    "    body_tokenized = [word for word in body_tokenized if word not in my_stopwords]\n",
    "\n",
    "    # Remove all punctuation\n",
    "    set_punctuation = {'?','.',',','!','@','#',\n",
    "                       '%','^','&','*','(',')','-',\n",
    "                       '{','}','[',']','\\\\',\"|\",':',\n",
    "                       ';','<','>','/','$','~','`','\"\"','\\'',\n",
    "                      '``',\"''\", \"—\",\"'\",\"’\",\"'s\"}\n",
    "    body_tokenized = [word for word in body_tokenized if word not in\n",
    "                     set_punctuation]\n",
    "    \n",
    "    # Turn back into a string\n",
    "    mystring = ''\n",
    "    for word in body_tokenized:\n",
    "        mystring += word + ' '\n",
    "    \n",
    "    return mystring"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean the 'Title' column into a new column\n",
    "IIb_df['title_cleaned'] = IIb_df['Title'].apply(lambda x: partially_clean_title(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clean the 'Body' column into a new column\n",
    "IIb_df['body_cleaned'] = IIb_df['Body'].apply(lambda x: partially_clean_body(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Concatenate the 'Body' and 'Title' text to make modeling easier\n",
    "IIb_df['text_cleaned'] = IIb_df['title_cleaned'] + ' ' + IIb_df['body_cleaned']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Section III: Feature Extraction and Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### III.a.: Extracting Features from, and Subsequently Modeling, the Fully Cleaned Dataset\n",
    "\n",
    "We will be comparing CountVectorizer and TFIDF features and performing 5-fold cross-validation in order to best assess the predictive power and performance of our models. We start by examining the fully cleaned dataset's performance and compare to an \"optimal\" dummy model. In section III.b., we will repeat this modeling process on an only partially-cleaned dataset to determine whether we may have cleaned excessively. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/joblib/externals/loky/process_executor.py:688: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.7119192238149952\n",
      "The ideal parameters are given by {'CountVector__max_features': 2000, 'CountVector__min_df': 10, 'CountVector__ngram_range': (1, 2), 'LogReg_cv__C': 1, 'LogReg_cv__penalty': 'l2'}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "# We begin with CountVectorizer features\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "\n",
    "# See https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "# for information on using the Pipeline in order to combine feature\n",
    "# extraction with GridSearchCV in order to get the optimal combination of\n",
    "# hyperparameters\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "# Split the data appropriately\n",
    "X = IIa_df['text_cleaned']\n",
    "y = y_train\n",
    "\n",
    "# Create a blank CountVectorizer object and model object (and pipeline)\n",
    "CountVector = CountVectorizer()\n",
    "LogReg_cv = LogisticRegression(random_state=0)\n",
    "pipeline_lr_cv = Pipeline([('CountVector',CountVector),('LogReg_cv',LogReg_cv)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_lr_cv = {'CountVector__ngram_range': [(1,2), (1,3)],\n",
    "             'CountVector__min_df': [5, 10],\n",
    "             'CountVector__max_features': [2000, 3000],\n",
    "             'LogReg_cv__C':[0.1, 1, 10],\n",
    "             'LogReg_cv__penalty':['l1','l2']}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_lr_cv = GridSearchCV(pipeline_lr_cv,\n",
    "                            parameters_lr_cv, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_lr_cv.fit(X,  y)\n",
    "\n",
    "best_logreg_f1_weighted_cv = my_gridsearch_lr_cv.best_score_\n",
    "best_logreg_parameters_cv = my_gridsearch_lr_cv.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_logreg_f1_weighted_cv)\n",
    "print('The ideal parameters are given by',\n",
    "     best_logreg_parameters_cv)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/joblib/externals/loky/process_executor.py:688: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.7149094073155702\n",
      "The ideal parameters are given by {'LogReg__C': 10, 'LogReg__penalty': 'l2', 'TfidfVector__max_features': 2000, 'TfidfVector__min_df': 5, 'TfidfVector__ngram_range': (1, 3)}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "# We conclude with Tfidf Features\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# See https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "# for information on using the Pipeline in order to combine TFIDF\n",
    "# modeling with GridSearchCV in order to get the optimal combination of\n",
    "# hyperparameters\n",
    "\n",
    "# Split the data appropriately\n",
    "X = IIa_df['text_cleaned']\n",
    "y = y_train\n",
    "\n",
    "# Create a blank TFIDF vector object and model object (and pipeline)\n",
    "TfidfVector = TfidfVectorizer()\n",
    "LogReg = LogisticRegression(random_state=0)\n",
    "pipeline_lr_tfidf = Pipeline([('TfidfVector',TfidfVector),('LogReg',LogReg)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_lr_tfidf = {'TfidfVector__ngram_range': [(1,2), (1,3)],\n",
    "             'TfidfVector__min_df': [5, 10],\n",
    "             'TfidfVector__max_features': [2000, 3000],\n",
    "             'LogReg__C':[0.1, 1, 10],\n",
    "             'LogReg__penalty':['l1','l2']}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_lr_tfidf = GridSearchCV(pipeline_lr_tfidf,\n",
    "                            parameters_lr_tfidf, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_lr_tfidf.fit(X,  y)\n",
    "\n",
    "best_logreg_f1_weighted_tfidf = my_gridsearch_lr_tfidf.best_score_\n",
    "best_logreg_parameters_tfidf = my_gridsearch_lr_tfidf.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_logreg_f1_weighted_tfidf)\n",
    "print('The ideal parameters are given by',\n",
    "     best_logreg_parameters_tfidf)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/joblib/externals/loky/process_executor.py:688: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.6872593288331049\n",
      "The ideal parameters are given by {'CountVector_rf__max_features': 2000, 'CountVector_rf__min_df': 5, 'CountVector_rf__ngram_range': (1, 3), 'RF_cv__max_depth': None, 'RF_cv__n_estimators': 50}\n"
     ]
    }
   ],
   "source": [
    "# Now, perform the same GridSearchCV except this time we use\n",
    "# a RandomForestClassifier instead. This is again based on\n",
    "# the sklearn paradigm at https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "\n",
    "# Create a blank CountVectorizer object and model object (and pipeline)\n",
    "CountVector_rf = CountVectorizer()\n",
    "RF_cv = RandomForestClassifier(random_state=0)\n",
    "pipeline_rf_cv = Pipeline([('CountVector_rf',CountVector_rf),('RF_cv',RF_cv)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_rf_cv = {'CountVector_rf__ngram_range': [(1,2), (1,3)],\n",
    "             'CountVector_rf__min_df': [5, 10],\n",
    "             'CountVector_rf__max_features': [2000, 3000],\n",
    "             'RF_cv__n_estimators':[50, 100, 200],\n",
    "             'RF_cv__max_depth':[5,10,None]}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_rf_cv = GridSearchCV(pipeline_rf_cv,\n",
    "                            parameters_rf_cv, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_rf_cv.fit(X,  y)\n",
    "\n",
    "best_rf_f1_weighted_cv = my_gridsearch_rf_cv.best_score_\n",
    "best_rf_parameters_cv = my_gridsearch_rf_cv.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_rf_f1_weighted_cv)\n",
    "print('The ideal parameters are given by',\n",
    "     best_rf_parameters_cv)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/joblib/externals/loky/process_executor.py:688: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.6828467287443758\n",
      "The ideal parameters are given by {'RF_tfidf__max_depth': None, 'RF_tfidf__n_estimators': 50, 'TfidfVector_rf__max_features': 2000, 'TfidfVector_rf__min_df': 10, 'TfidfVector_rf__ngram_range': (1, 3)}\n"
     ]
    }
   ],
   "source": [
    "# Now, perform the same GridSearchCV except this time we use\n",
    "# a RandomForestClassifier instead. This is again based on\n",
    "# the sklearn paradigm at https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "\n",
    "# Create a blank TFIDF vector object and model object (and pipeline)\n",
    "TfidfVector_rf = TfidfVectorizer()\n",
    "RF_tfidf = RandomForestClassifier(random_state=0)\n",
    "pipeline_rf_tfidf = Pipeline([('TfidfVector_rf',TfidfVector_rf),('RF_tfidf',RF_tfidf)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_rf_tfidf = {'TfidfVector_rf__ngram_range': [(1,2), (1,3)],\n",
    "             'TfidfVector_rf__min_df': [5, 10],\n",
    "             'TfidfVector_rf__max_features': [2000, 3000],\n",
    "             'RF_tfidf__n_estimators':[50, 100, 200],\n",
    "             'RF_tfidf__max_depth':[5,10,None]}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_rf_tfidf = GridSearchCV(pipeline_rf_tfidf,\n",
    "                            parameters_rf_tfidf, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_rf_tfidf.fit(X,  y)\n",
    "\n",
    "best_rf_f1_weighted_tfidf = my_gridsearch_rf_tfidf.best_score_\n",
    "best_rf_parameters_tfidf = my_gridsearch_rf_tfidf.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_rf_f1_weighted_tfidf)\n",
    "print('The ideal parameters are given by',\n",
    "     best_rf_parameters_tfidf)    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Compare briefly the better model with the best-possible dummy model. This model technically may use slightly different TFIDF features, but nonetheless, it should not exceed the better of the two previous models (or approximately equal the better of the two previous models), since that would imply that our models are not really learning anything from the data. TFIDF performed the best above, so we will just compare with TFIDF features and dummy classifier. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/joblib/externals/loky/process_executor.py:688: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.6731629010041636\n",
      "The ideal parameters are given by {'Dummy__strategy': 'most_frequent', 'TfidfVector__max_features': 2000, 'TfidfVector__min_df': 5, 'TfidfVector__ngram_range': (1, 2)}\n"
     ]
    }
   ],
   "source": [
    "# Now, perform the same GridSearchCV except this time we use\n",
    "# a DummyClassifier instead. This is again based on\n",
    "# the sklearn paradigm at https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "from sklearn.dummy import DummyClassifier\n",
    "\n",
    "# Create a blank TFIDF vector object and model object (and pipeline)\n",
    "TfidfVector = TfidfVectorizer()\n",
    "Dummy = DummyClassifier()\n",
    "pipeline_dummy = Pipeline([('TfidfVector',TfidfVector),('Dummy',Dummy)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_dummy = {'TfidfVector__ngram_range': [(1,2), (1,3)],\n",
    "             'TfidfVector__min_df': [5, 10],\n",
    "             'TfidfVector__max_features': [2000, 3000],\n",
    "             'Dummy__strategy':['stratified','most_frequent',\n",
    "                               'uniform']}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_dummy = GridSearchCV(pipeline_dummy,\n",
    "                            parameters_dummy, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_dummy.fit(X,  y)\n",
    "\n",
    "best_dummy_f1_weighted = my_gridsearch_dummy.best_score_\n",
    "best_dummy_parameters = my_gridsearch_dummy.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_dummy_f1_weighted)\n",
    "print('The ideal parameters are given by',\n",
    "     best_dummy_parameters)    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Note that our best model (with weighted f1-score of 0.714909) using the fully cleaned dataset outperforms the dummy model (with weighted f1-score of 0.673163) sufficiently to know that it is learning at least something from the data. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### III.b.: Extracting Features from, and Subsequently Modeling, the Partially Cleaned Dataset\n",
    "\n",
    "We now repeat the modeling process above on an only partially-cleaned dataset to determine whether we may have excessively cleaned the data initially."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/joblib/externals/loky/process_executor.py:688: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.7126187981756603\n",
      "The ideal parameters are given by {'CountVector_b__max_features': 2000, 'CountVector_b__min_df': 5, 'CountVector_b__ngram_range': (1, 2), 'LogReg_cv_b__C': 1, 'LogReg_cv_b__penalty': 'l2'}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "# We begin with CountVectorizer features\n",
    "\n",
    "# See https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "# for information on using the Pipeline in order to combine feature\n",
    "# extraction with GridSearchCV in order to get the optimal combination of\n",
    "# hyperparameters\n",
    "\n",
    "# Split the data appropriately\n",
    "X_b = IIb_df['text_cleaned']\n",
    "y_b = y_train\n",
    "\n",
    "\n",
    "# Create a blank CountVectorizer object and model object (and pipeline)\n",
    "CountVector_b = CountVectorizer()\n",
    "LogReg_cv_b = LogisticRegression(random_state=0)\n",
    "pipeline_lr_cv_b = Pipeline([('CountVector_b',CountVector_b),('LogReg_cv_b',LogReg_cv_b)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_lr_cv_b = {'CountVector_b__ngram_range': [(1,2), (1,3)],\n",
    "             'CountVector_b__min_df': [5, 10],\n",
    "             'CountVector_b__max_features': [2000, 3000],\n",
    "             'LogReg_cv_b__C':[0.1, 1, 10],\n",
    "             'LogReg_cv_b__penalty':['l1','l2']}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_lr_cv_b = GridSearchCV(pipeline_lr_cv_b,\n",
    "                            parameters_lr_cv_b, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_lr_cv_b.fit(X_b,  y_b)\n",
    "\n",
    "best_logreg_f1_weighted_cv_b = my_gridsearch_lr_cv_b.best_score_\n",
    "best_logreg_parameters_cv_b = my_gridsearch_lr_cv_b.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_logreg_f1_weighted_cv_b)\n",
    "print('The ideal parameters are given by',\n",
    "     best_logreg_parameters_cv_b)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.7156493458466487\n",
      "The ideal parameters are given by {'LogReg_b__C': 10, 'LogReg_b__penalty': 'l2', 'TfidfVector_b__max_features': 2000, 'TfidfVector_b__min_df': 10, 'TfidfVector_b__ngram_range': (1, 2)}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "# We conclude with Tfidf Features\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# See https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "# for information on using the Pipeline in order to combine TFIDF\n",
    "# modeling with GridSearchCV in order to get the optimal combination of\n",
    "# hyperparameters\n",
    "\n",
    "# Split the data appropriately\n",
    "X_b = IIb_df['text_cleaned']\n",
    "y_b = y_train\n",
    "\n",
    "# Create a blank TFIDF vector object and model object (and pipeline)\n",
    "TfidfVector_b = TfidfVectorizer()\n",
    "LogReg_b = LogisticRegression(random_state=0)\n",
    "pipeline_lr_tfidf_b = Pipeline([('TfidfVector_b',TfidfVector_b),('LogReg_b',LogReg_b)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_lr_tfidf_b = {'TfidfVector_b__ngram_range': [(1,2), (1,3)],\n",
    "             'TfidfVector_b__min_df': [5, 10],\n",
    "             'TfidfVector_b__max_features': [2000, 3000],\n",
    "             'LogReg_b__C':[0.1, 1, 10],\n",
    "             'LogReg_b__penalty':['l1','l2']}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_lr_tfidf_b = GridSearchCV(pipeline_lr_tfidf_b,\n",
    "                            parameters_lr_tfidf_b, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_lr_tfidf_b.fit(X_b,  y_b)\n",
    "\n",
    "best_logreg_f1_weighted_tfidf_b = my_gridsearch_lr_tfidf_b.best_score_\n",
    "best_logreg_parameters_tfidf_b = my_gridsearch_lr_tfidf_b.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_logreg_f1_weighted_tfidf_b)\n",
    "print('The ideal parameters are given by',\n",
    "     best_logreg_parameters_tfidf_b)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/joblib/externals/loky/process_executor.py:688: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.6873279446644502\n",
      "The ideal parameters are given by {'CountVector_rf_b__max_features': 2000, 'CountVector_rf_b__min_df': 5, 'CountVector_rf_b__ngram_range': (1, 3), 'RF_cv_b__max_depth': None, 'RF_cv_b__n_estimators': 50}\n"
     ]
    }
   ],
   "source": [
    "# Now, perform the same GridSearchCV except this time we use\n",
    "# a RandomForestClassifier instead. This is again based on\n",
    "# the sklearn paradigm at https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "\n",
    "# Create a blank CountVectorizer object and model object (and pipeline)\n",
    "CountVector_rf_b = CountVectorizer()\n",
    "RF_cv_b = RandomForestClassifier(random_state=0)\n",
    "pipeline_rf_cv_b = Pipeline([('CountVector_rf_b',CountVector_rf_b),('RF_cv_b',RF_cv_b)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_rf_cv_b = {'CountVector_rf_b__ngram_range': [(1,2), (1,3)],\n",
    "             'CountVector_rf_b__min_df': [5, 10],\n",
    "             'CountVector_rf_b__max_features': [2000, 3000],\n",
    "             'RF_cv_b__n_estimators':[50, 100, 200],\n",
    "             'RF_cv_b__max_depth':[5,10,None]}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_rf_cv_b = GridSearchCV(pipeline_rf_cv_b,\n",
    "                            parameters_rf_cv_b, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_rf_cv_b.fit(X_b,  y_b)\n",
    "\n",
    "best_rf_f1_weighted_cv_b = my_gridsearch_rf_cv_b.best_score_\n",
    "best_rf_parameters_cv_b = my_gridsearch_rf_cv_b.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_rf_f1_weighted_cv_b)\n",
    "print('The ideal parameters are given by',\n",
    "     best_rf_parameters_cv_b)    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/joblib/externals/loky/process_executor.py:688: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.683423226071391\n",
      "The ideal parameters are given by {'RF_tfidf_b__max_depth': None, 'RF_tfidf_b__n_estimators': 50, 'TfidfVector_rf_b__max_features': 2000, 'TfidfVector_rf_b__min_df': 5, 'TfidfVector_rf_b__ngram_range': (1, 2)}\n"
     ]
    }
   ],
   "source": [
    "# Now, perform the same GridSearchCV except this time we use\n",
    "# a RandomForestClassifier instead. This is again based on\n",
    "# the sklearn paradigm at https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "\n",
    "# Create a blank TFIDF vector object and model object (and pipeline)\n",
    "TfidfVector_rf_b = TfidfVectorizer()\n",
    "RF_tfidf_b = RandomForestClassifier(random_state=0)\n",
    "pipeline_rf_tfidf_b = Pipeline([('TfidfVector_rf_b',TfidfVector_rf_b),('RF_tfidf_b',RF_tfidf_b)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_rf_tfidf_b = {'TfidfVector_rf_b__ngram_range': [(1,2), (1,3)],\n",
    "             'TfidfVector_rf_b__min_df': [5, 10],\n",
    "             'TfidfVector_rf_b__max_features': [2000, 3000],\n",
    "             'RF_tfidf_b__n_estimators':[50, 100, 200],\n",
    "             'RF_tfidf_b__max_depth':[5,10,None]}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_rf_tfidf_b = GridSearchCV(pipeline_rf_tfidf_b,\n",
    "                            parameters_rf_tfidf_b, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_rf_tfidf_b.fit(X_b,  y_b)\n",
    "\n",
    "best_rf_f1_weighted_tfidf_b = my_gridsearch_rf_tfidf_b.best_score_\n",
    "best_rf_parameters_tfidf_b = my_gridsearch_rf_tfidf_b.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_rf_f1_weighted_tfidf_b)\n",
    "print('The ideal parameters are given by',\n",
    "     best_rf_parameters_tfidf_b)    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Compare briefly the better model with the best-possible dummy model. This model technically may use slightly different TfidfVectorizer features, but nonetheless, it should not exceed the better of the two previous models (or approximately equal the better of the two previous models), since that would imply that our models are not really learning anything from the data. TFIDF performed the best above, so we will just compare with a dummy classifier using TFIDF features alone."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/joblib/externals/loky/process_executor.py:688: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The best weighted f1-score is given by 0.6731629010041636\n",
      "The ideal parameters are given by {'Dummy__strategy': 'most_frequent', 'TfidfVector__max_features': 2000, 'TfidfVector__min_df': 5, 'TfidfVector__ngram_range': (1, 2)}\n"
     ]
    }
   ],
   "source": [
    "# Now, perform the same GridSearchCV except this time we use\n",
    "# a DummyClassifier instead. This is again based on\n",
    "# the sklearn paradigm at https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html\n",
    "from sklearn.dummy import DummyClassifier\n",
    "\n",
    "# Create a blank TFIDF vector object and model object (and pipeline)\n",
    "TfidfVectorDummy2 = TfidfVectorizer()\n",
    "Dummy2 = DummyClassifier()\n",
    "pipeline_dummy2 = Pipeline([('TfidfVector',TfidfVector),('Dummy',Dummy)])\n",
    "\n",
    "# Denote the parameters which we are interested in testing\n",
    "parameters_dummy2 = {'TfidfVector__ngram_range': [(1,2), (1,3)],\n",
    "             'TfidfVector__min_df': [5, 10],\n",
    "             'TfidfVector__max_features': [2000, 3000],\n",
    "             'Dummy__strategy':['stratified','most_frequent',\n",
    "                               'uniform']}\n",
    "\n",
    "# Perform a GridSearch for the best model. We use the \n",
    "# weighted f1-score to determine how good our model is\n",
    "my_gridsearch_dummy2 = GridSearchCV(pipeline_dummy2,\n",
    "                            parameters_dummy2, n_jobs = -1,\n",
    "                            scoring = 'f1_weighted')\n",
    "\n",
    "my_gridsearch_dummy2.fit(X_b,  y_b)\n",
    "\n",
    "best_dummy_f1_weighted2 = my_gridsearch_dummy2.best_score_\n",
    "best_dummy_parameters2 = my_gridsearch_dummy2.best_params_\n",
    "\n",
    "# Print out the results of the best model so that we can\n",
    "# directly apply it\n",
    "print('The best weighted f1-score is given by',\n",
    "     best_dummy_f1_weighted2)\n",
    "print('The ideal parameters are given by',\n",
    "     best_dummy_parameters2)    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Note that once again, our best model (with weighted f1-score of 0.715649) using the only partially cleaned dataset outperforms the dummy model (with weighted f1-score of 0.673163) sufficiently to know that it is learning at least something from the data. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Therefore, our best model as evidenced by cross-validation performance on the training set is given by these objects (applied to the partially cleaned dataset):\n",
    "\n",
    "For feature extraction, we use TfidfVectorizer(max_features=2000, min_df=10, ngram_range=(1, 2)). \n",
    "\n",
    "For the model itself to be fitted, we use LogisticRegression(random_state=0, C=10, penalty='l2').\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Section IV: Test Set Performance"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Note that the optimal combination, as decided based upon training set cross-validation performance, is as follows, with the optimal cleaning pattern described as partially cleaning the data:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "#### Clean the test dataset properly ####\n",
    "\n",
    "# Clean the 'Title' column into a new column\n",
    "X_test['title_cleaned'] = X_test['Title'].apply(lambda x: partially_clean_title(x))\n",
    "\n",
    "# Clean the 'Body' column into a new column\n",
    "X_test['body_cleaned'] = X_test['Body'].apply(lambda x: partially_clean_body(x))\n",
    "\n",
    "# Concatenate the 'Body' and 'Title' text to make modeling easier\n",
    "X_test['text_cleaned'] = X_test['title_cleaned'] + ' ' + X_test['body_cleaned']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Featurize the test dataset properly\n",
    "\n",
    "# Split the data appropriately\n",
    "X_test_revs = X_test['text_cleaned']\n",
    "y_test = y_test\n",
    "\n",
    "# Create a blank TfidfVectorizer object to prepare to featurize\n",
    "# the data\n",
    "TfidfVector_test = TfidfVectorizer(max_features=2000, min_df=10, ngram_range=(1, 2))\n",
    "TfidfVector_test.fit(X_b) #X_b was the optimal partially-cleaned dataset of only training data\n",
    "X_train_vectorized = TfidfVector_test.transform(X)\n",
    "X_test_vectorized = TfidfVector_test.transform(X_test_revs)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "# Fit the best model on the entire training dataset and make \n",
    "# predictions on the test set accordingly\n",
    "\n",
    "best_model = LogisticRegression(random_state=0, C=10, penalty='l2')\n",
    "best_model.fit(X_train_vectorized, y_train)\n",
    "y_test_pred = best_model.predict(X_test_vectorized)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.41      0.16      0.23      1256\n",
      "           1       0.80      0.94      0.86      4483\n",
      "\n",
      "    accuracy                           0.77      5739\n",
      "   macro avg       0.60      0.55      0.54      5739\n",
      "weighted avg       0.71      0.77      0.72      5739\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Score the best model\n",
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test, y_test_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.00      0.00      0.00      1256\n",
      "           1       0.78      1.00      0.88      4483\n",
      "\n",
      "    accuracy                           0.78      5739\n",
      "   macro avg       0.39      0.50      0.44      5739\n",
      "weighted avg       0.61      0.78      0.69      5739\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Clayton/opt/anaconda3/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1221: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n"
     ]
    }
   ],
   "source": [
    "# Score the best model against the dummy classifier's performance\n",
    "dummy_test = DummyClassifier(strategy='most_frequent')\n",
    "dummy_test.fit(X_train_vectorized, y_train)\n",
    "y_test_pred_dummy = dummy_test.predict(X_test_vectorized)\n",
    "print(classification_report(y_test, y_test_pred_dummy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Note, then, that our best model achieves a weighted f1-score of 0.72 on the test dataset, while the best possible dummy classifier (from our given grid search on the training data in cross-validation) achieves a score of only 0.69. So, in this sense, we significantly beat the baseline models, although the problem of achieving a very high f1-score on this dataset—and of predicting whether a question might be positively or negatively scored—appears to be quite difficult. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Section V: Comparing Performance via a Precision-Recall Curve"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot the precision-recall curve, per the lecture notes from Lecture 13\n",
    "from sklearn.metrics import precision_recall_curve\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Predict the probabilities with our best model\n",
    "y_test_pred_prob = best_model.predict_proba(X_test_vectorized)[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Precision')"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Generate the precision-recall curve\n",
    "precision, recall, threshold_vals = precision_recall_curve(y_test, y_test_pred_prob)\n",
    "plt.plot(recall, precision)\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Notice that the precision still remains high (above 0.7 uniformly, and mostly above 0.8) on the y-axis even as the recall shifts from 0 to 1. Therefore, out of all of the questions classified as positive questions, even as we increase the recall (which is the proportion of truly positive questions that are classified as positive) to 1, the proportion of the positively-classified questions that are truly positive remains high. "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
