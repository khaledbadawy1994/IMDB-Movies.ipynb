# IMDB-Movies.ipynb

What is IMDb ?

IMDb, short for Internet Movie Database, is an online repository that houses a vast collection of information related to movies, TV shows, podcasts, home videos, video games, and streaming content. It provides details such as cast and crew information, personal biographies, plot summaries, trivia, ratings, as well as fan and critical reviews. Initially established by fans on the Usenet group "rec.arts.movies" in 1990, IMDb transitioned to the internet in 1993. Since 1998, it has been under the ownership and operation of IMDb.com, Inc., a subsidiary of Amazon.

As of March 2022, the database encompasses approximately 10.1 million titles, including television episodes, and boasts 11.5 million records of individuals. Furthermore, the site has amassed an impressive user base of 83 million registered users. The platform's message boards were disabled in February 2017.¶

Outline of EDA We shall perform the following steps:

Preview data
Check total number of entries and column types
Check any null values
Check duplicate entries
Rename the columns
Formatting data
Create any additional data
Plot findings
import pandas as pdimport numpy as npimport matplotlib.pyplot as pltimport matplotlib as mplimport seaborn as snsimport datetime as dtimport pylab as pl%matplotlib inline

# Loading the dataset

Movies = pd.read_csv("/content/drive/MyDrive/imdb_movies.csv")
Movies.head()

Movies.tail()

Movies.rename(columns = {'DATE_X':'DATE', 'BUDGET_X': 'BUDGET'}, inplace = True)

Movies.columns = [x.upper() for x in Movies.columns]
Movies.describe().transpose()

# statistical summary of all numeric-typed (int, float) columns
print("\nDataset Summary:")
Movies.describe()

# statistical summary of all columns
print("\nDataset Summary:")
Movies.describe(include="all")

# get the list of column headers
print("\nColumn Names:")
Movies.columns

# Number of Rows
print("Number of rows",Movies.shape[0])

# Number of colums
print("number of colums",Movies.shape[1])

Movies.duplicated()

# to find duplicate values in  "orig_title"column

duplicate=Movies[Movies.duplicated("ORIG_TITLE")]
duplicate["ORIG_TITLE"].value_counts()

# drop that duplicate values row
Movies.drop_duplicates(subset="ORIG_TITLE",inplace=True)
Cleaning the Dataframe

1. Identifying Missing Values

# Two methods to detect missing data are df.isnull() or df.isnotnull()
missing_data= Movies.isnull()
missing_data.head()

Movies.isna().sum()

Movies.info()

# the column "crew" is not required for any analysis, so we drop it.
Movies.drop(['CREW'], axis=1, inplace=True)
Movies["DATE"]= pd.to_datetime(Movies["DATE"])
# the column "genre" has some missing values which can be replaced by frequency

# see which values are present in column "genre"
Movies['GENRE'].value_counts()

# another method to check the most common value in the column is by using ".idxmax()"
Movies['GENRE'].value_counts().idxmax()

Movies.columns

Movies.COUNTRY.unique()

Movies.dropna(inplace=True)
Movies.shape

Movies.describe().T.style.format('{:,.2F}')

matrix=Movies[["REVENUE","SCORE","BUDGET"]].corr()
matrix.style.background_gradient(cmap='Reds', axis=0)

plt.figure(figsize=(10,8))
plt.title('Correlation between Revenue vs Score vs Budget')
corr = Movies[["REVENUE","SCORE","BUDGET"]].corr()
sns.heatmap(corr, annot=True)

top_10_revenues = Movies.sort_values("REVENUE",ascending=True)[-10:]
top_10_revenues

Movies.sort_values(by=['ORIG_TITLE','COUNTRY', 'ORIG_LANG'], ascending=[True, True, True])
Movies[['ORIG_TITLE','COUNTRY', 'ORIG_LANG']]

Movies['ORIG_LANG'].value_counts()

Movies['COUNTRY'].value_counts()

Movies['REVENUE'].nlargest()

Movies['REVENUE'].nsmallest()

country_grp = Movies.groupby(['COUNTRY'])
country_grp.get_group('US')

country_grp['ORIG_LANG'].value_counts().loc['US']

country_grp[['BUDGET', 'REVENUE']].agg(['median', 'mean'])

country_grp['ORIG_LANG'].apply(lambda x: x.str.contains('English').sum())

country_respondents = Movies['COUNTRY'].value_counts()

country_on_la = country_grp['ORIG_LANG'].apply(lambda x: x.str.contains('English').sum())

python_df = pd.concat([country_respondents, country_on_la],axis='columns', sort=True)
python_df.sort_values(by='COUNTRY', ascending=False, inplace=True)
python_df

# Now let us check which are the top 10 movies which generated highest revenue

# create a data frame, sort and format values
data = pd.DataFrame(Movies, columns=['REVENUE', 'NAMES'])
data_sorted = data.sort_values(by='REVENUE', ascending=False)
data_sorted['REVENUE'] = data_sorted['REVENUE'] / 1000000
pd.options.display.float_format = '{:,.0f}'.format
data_sorted.set_index('NAMES', inplace=True)
ranking_rev = data_sorted.head(10)
ranking_rev

top_10_revenue = Movies.sort_values('REVENUE', ascending=True)[-10:]
top_10_revenue

Data Vizualisation

# Variables
index = ranking_rev.index
values = ranking_rev['REVENUE']
plot_title = 'Top 10 movies by revenue, USD million'
title_size = 18
subtitle = 'Source: Kaggle / IMDB Movies'
x_label = 'REVENUE, USD million'
filename = 'barh-plot'
# draw a figure with a subplot. We’re using the viridis color scheme to create gradients later.
fig, ax = plt.subplots(figsize=(10,6), facecolor=(.94, .94, .94))
mpl.pyplot.viridis()

# create bars
bar = ax.barh(index, values, color='darkseagreen')
plt.tight_layout()
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# set title, its font size, and position
title = plt.title(plot_title, pad=20, fontsize=title_size)
title.set_position([.33, 1])
plt.subplots_adjust(top=0.9, bottom=0.1)

# create bar labels/annotations
rects = ax.patches
# Place a label for each bar
for rect in rects:
    # Get X and Y placement of label from rect
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label; change to your liking
    space = -30
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: place label to the left of the bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label to the right
        ha = 'right'

    # Use X value as label and format number
    label = '{:,.0f}'.format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at bar end
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords='offset points', # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha,                      # Horizontally align label differently for positive and negative values
        color = 'black')            # Change label color to white

   # Set subtitle
    tfrom = ax.get_xaxis_transform()
    ann = ax.annotate(subtitle, xy=(5, 1), xycoords=tfrom, bbox=dict(boxstyle='square,pad=1.3', fc='#f0f0f0', ec='none'))

   #Set x-label
    ax.set_xlabel(x_label, color='black')

# Now let us check which are the top 10 movies which generated highest revenue

# create a data frame, sort and format values

revenue_by_country =Movies.loc[:, ['COUNTRY', 'REVENUE']]
revenue_by_country = revenue_by_country.groupby('COUNTRY').sum()
top_5_countries = revenue_by_country.nlargest(5, 'REVENUE')/ 1000000
pd.options.display.float_format = '{:,.0f}'.format
# Variables
index = top_5_countries.index
values = top_5_countries['REVENUE']
plot_title = 'Top 05 Countries by revenue, USD million'
title_size = 18
subtitle = 'Source: Kaggle / IMDB Movies'
x_label = 'Revenue, USD million'
# Create a bar plot

fig, ax = plt.subplots(figsize=(15,6), facecolor=(.94, .94, .94))
mpl.pyplot.viridis()

# create bars
bar = ax.barh(index, values, color='darkseagreen')
plt.tight_layout()
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

# set title, its font size, and position
title = plt.title(plot_title, pad=20, fontsize=title_size)
title.set_position([.33, 1])
plt.subplots_adjust(top=0.9, bottom=0.1)

# create bar labels/annotations
rects = ax.patches
# Place a label for each bar
for rect in rects:
    # Get X and Y placement of label from rect
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label; change to your liking
    space = -25
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: place label to the left of the bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label to the right
        ha = 'right'

    # Use X value as label and format number
    label = '{:,.0f}'.format(x_value)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at bar end
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords='offset points', # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha='center',                      # Horizontally align label differently for positive and negative values
        color = 'black')            # Change label color to white

# Show the plot
plt.show()

# Now we compare the Budget against the Revenue generated for the Top 10 movies with highest revenue

# Sort the DataFrame by revenue in descending order and select the top 10 movies
top_10_movies = Movies.sort_values('REVENUE', ascending=False).head(10)
top_10_movies.set_index('NAMES', inplace=True)
top_10_movies

# now we keep only the columns budget_x and revenue and remove all other columns from the dataframe
# we also remove any duplicate values present in the dataframe

top_10_movies = top_10_movies[['BUDGET','REVENUE']]
top_10_movies = top_10_movies.drop_duplicates()
top_10_movies

# now we plot a horizontal bar chart comparing the revenue generated against the budget for the top 10 movies

ax = top_10_movies.plot.barh(width=0.8, color=['darkseagreen', 'deepskyblue'])

custom_legend = ['BUDGET', 'REVENUE']
ax.legend(custom_legend)
ax.set_xlabel("Amount in Millions")
ax.set_ylabel("Top 10 Movies")
ax.set_title("Budget against Revenue Generated for Top 10 Movies")

Movies['income_mil'] = (Movies['REVENUE'] - Movies['BUDGET']) #creating a new P/L column
Movies['DATE'] = pd.to_datetime(Movies['DATE'])
Movies['YEAR'] = Movies['DATE'].dt.year #creating a new year column
# now we check which are the top 5 languages of the movies which generate maximum revenue

rev_by_lang = Movies.loc[:, ['ORIG_LANG', 'REVENUE']]
rev_by_lang = rev_by_lang.groupby('ORIG_LANG').sum()
top_5_lang = rev_by_lang.nlargest(5, 'REVENUE')
top_5_lang

plt.figure(figsize=(15,6))
# Create a bar plot
plt.barh(top_5_lang.index, top_5_lang['REVENUE']/1000000000, color='darkseagreen')
plt.title('Top 5 Languages of the Movies Generating Maximum Revenue', fontsize=20)

# Label the axes
plt.ylabel('LANGUAGE', fontsize=15)
plt.xlabel('Revenue (in billions)', fontsize=15)

# Show the plot
plt.show()

country_count = Movies['COUNTRY'].value_counts().head(10)

plt.bar(country_count.index, height=country_count)
plt.show()

genre_count= Movies['GENRE'].value_counts().head(10)
genre_count

plt.figure(figsize=(15,6))
# Create a bar plot
plt.bar(top_10_revenues['NAMES'], top_10_revenues['REVENUE'])

# Set the x-axis labels
plt.xticks(rotation=60)

plt.title('Top 10 Movie Revenues')
# Label the axes
plt.xlabel('Movie Title')
plt.ylabel('Revenue')

# Show the plot
plt.show()

revenue_by_country = Movies.loc[:, ['COUNTRY', 'REVENUE']]
revenue_by_country = revenue_by_country.groupby('COUNTRY').sum()
top_5_countries = revenue_by_country.nlargest(5, 'REVENUE')

plt.figure(figsize=(15,6))
# Create a bar plot
plt.bar(top_5_countries.index, top_5_countries['REVENUE']/1000000000)
plt.title('Top 5 Countries with Highest Revenue')

# Set the x-axis labels
plt.xticks(rotation=60)

# Label the axes
plt.xlabel('Country')
plt.ylabel('Revenue in billions')

# Show the plot
plt.show()

# now we check which is the most common language in which the movies are produced

orig_lang_count= Movies['ORIG_LANG'].value_counts().head(10)
orig_lang_count

revenue_by_lang = Movies.loc[:, ['ORIG_LANG', 'REVENUE']]
revenue_by_lang = revenue_by_lang.groupby('ORIG_LANG').sum()
top_5_langs = revenue_by_lang.nlargest(5, 'REVENUE')

plt.figure(figsize=(15,6))
# Create a bar plot
plt.bar(top_5_langs.index, top_5_langs['REVENUE']/1000000000)
plt.title('Top 5 Language Spoken Movies')

# Set the x-axis labels
plt.xticks(rotation=60)

# Label the axes
plt.xlabel('Country')
plt.ylabel('Revenue (in billions)')

# Show the plot
plt.show()

y= Movies["ORIG_LANG"].value_counts().index.tolist()[0:9]
y

x=Movies["ORIG_LANG"].value_counts().unique()[0:9]
x

# plot a pie chart
plt.figure(figsize=(8,8))
plt.pie(orig_lang_count, labels=orig_lang_count.index, autopct="%0.01f%%")
plt.title('Top 10 Languages in which the movies are Produced')
plt.show()

v= Movies["GENRE"].value_counts().index.tolist()[0:9]
v

w=Movies["GENRE"].value_counts().unique()[0:9]
w

# now we check which is the most common Genre of movies produced

genre_count= Movies['GENRE'].value_counts().head(10)
genre_count

# plot a pie chart
plt.figure(figsize=(8,8))
plt.pie(genre_count, labels=genre_count.index, autopct="%0.01f%%")
plt.title('Top 10 Genre of Movies Produced')
plt.show()

orig_lang_conut = Movies['ORIG_LANG'].value_counts().head(10)
orig_lang_conut

plt.figure(figsize=(10, 10))

plt.title("Language in Movies")

plt.pie(orig_lang_conut, labels=orig_lang_conut.index,autopct="%0.0f%%")

plt.show()

score_conut = Movies['SCORE'].value_counts().head(10)
plt.pie(score_conut, labels=score_conut.index,autopct="%0.0f%%")

plt.show()

# now we check the number of movies released per year

# convert the date_x column from string format to date format
Movies["DATE"]= pd.to_datetime(Movies["DATE"])

# Get the count of movies released per year
year_counts = Movies["DATE"].dt.year.value_counts().sort_index()
# Split year and month in seperate columns
Movies["Release_year"]=Movies["DATE"].dt.year
Movies["Release_month"]=Movies["DATE"].dt.month
Movies['DATE'].dt.year.value_counts().sort_index()

# Generate the plot
plt.figure(figsize=(12,8))
plt.plot(year_counts.index, year_counts)

# Set the X-axis tick labels to show bins every 10 years
xticks = [year for year in year_counts.index if year % 5 == 0]
plt.xticks(xticks, rotation=45)

# Set the title of the plot
plt.title("Number of Movies Released per Year")
plt.xlabel("Years")
plt.ylabel("Counts")

# Show the plot
plt.show()

months=["jan","feb","mar","apr","may","june","july","aug","sept","oct","nov","dec"]
Movies["DATE"].dt.month.value_counts().sort_index()

# plot  between year and number of movies

colors = plt.cm.viridis(np.linspace(-3, 1, 40))  #to create some color combinations
plt.figure(figsize=(15,7))
Movies.groupby(pd.cut(Movies["Release_year"],bins=40))["NAMES"].agg("count").plot(kind="bar",color=colors)
plt.title("No of Movies release in a year")
plt.ylabel("movie count")
plt.show()

Result : In years basics the movies count is increasing faster

#plot between month and number of movies
plt.figure(figsize=(15,6))

Movies.groupby("Release_month")["NAMES"].agg("count").plot(kind="bar",color=colors)

#data.groupby("Release_month")["names"].agg("count").plot(kind="line",color="red")

plt.title("No of Movies release in a month")
plt.ylabel("movie count")
plt.show()

# now we check the number of movies released per month

months=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
month_count = Movies["DATE"].dt.month.value_counts().sort_index()
plt.figure(figsize=(12,8))
plt.bar(x=months, height= month_count, color = "darkseagreen")
plt.title("Total Number of Movies Released per Month from 1900 to 2023")
# Add values on top of each bar
for i, v in enumerate(month_count):
    plt.text(i, v, str(v), ha='center', va='bottom')

Result : The movie count is less in "JUNE ,JULY, AUGUST "Month -------->¶

More movies are released in "october" month

# plot between no of movies in their orgin language
plt.figure(figsize=(10,5))
Movies.groupby("ORIG_LANG")["NAMES"].agg("count").plot(kind="bar")
plt.title("No of Movies in Orign language")
plt.ylabel("movie count")
plt.show()

Result :¶ There are more number of movie released in "English" language as orgin Remaining count are less then 1000.

# plot depend on score
plt.figure(figsize=(10,5))
Movies.groupby("SCORE")["NAMES"].agg("count").plot(kind="line")
plt.title("score and their Movie count")
plt.ylabel("movie count")
plt.show()

Result :

The average movie score between 55 - 75

#import the module for remove the "le6" at y axis and formate it

from matplotlib.ticker import ScalarFormatter
plt.figure(figsize=(18,9))
colors = plt.cm.viridis(np.linspace(-3, 1, 60))
plt.bar(Movies["SCORE"],Movies["REVENUE"],color=colors)

plt.ticklabel_format(axis='y', style='plain')
plt.title("Revenue depend on score value")
plt.ylabel("Revenue")
plt.show()

Result :

The high revenue movies are range between 75-85 score

# plot depends  on country and movies
plt.figure(figsize=(10,5))
Movies.groupby("COUNTRY")["NAMES"].agg("count").plot(kind="bar",color="red")
plt.title("No of movie Released in a country")
plt.ylabel("Movie count")
plt.show()

Result :¶ There are more movie release in "1) AUSTRALIA ( AU ) ---- 2) UNITED STATES( US )

# plot the score  with orign language

from matplotlib.ticker import ScalarFormatter
plt.figure(figsize=(10,5))

sns.barplot(Movies,x=Movies["ORIG_LANG"],y=Movies["SCORE"])
plt.xticks(rotation=90)
plt.title("orign language with max score")
plt.ylabel("score")
plt.show()

Result :

The average score for the orign language : it denote [ "Tamil, malayalam, Hungarian ] has less score value

Movies['primary_genre'] = Movies['GENRE'].apply(lambda x: x.split(',')[0]) #creating a primary genre column
Movies['budget_tier'] = Movies['BUDGET'].apply(lambda x: "Low" if x < 5 \
                                             else "Mid" if 5 <= x <= 50 \
                                             else "High" if 50 < x <= 150 \
                                             else "Blockbuster") #creating a column with budget tier for every row
Movies['profitable'] = Movies['income_mil'].apply(lambda x: "Yes" if x > 0 else "No")
Movies.describe()

fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(8,6))
his1= Movies['SCORE']
his2= Movies['BUDGET']
his3= Movies['REVENUE']
his4= Movies['YEAR']
his5= Movies['income_mil']

axes[0,0].hist(his3, bins=50)
axes[0,0].set_title('Revenue')
axes[0,1].hist(his2, bins=50)
axes[0,1].set_title('Budget')
axes[1,0].hist(his1, bins=50)
axes[1,0].set_title('IMBD Rating')
axes[1,1].hist(his4, bins=50)
axes[1,1].set_title('Year Released')
axes[2,0].hist(his5, bins=50)
axes[2,0].set_title('Income')

fig.delaxes(axes[2, 1])

plt.tight_layout()
plt.show()

By examining the maximum values and the highly skewed distributions, I deduce that the Interquartile Range (IQR) would be the most suitable measure of variability for a dataset with outliers. Since 'budget_mil,' 'revenue_mil,' and 'income_mil' are interrelated, I will focus on revenue grouped by budget tiers rather than income.

Upon analyzing the boxplots, it becomes evident that movies with high and blockbuster budgets have a wider spread and box (representing 50% of the values), while those with mid and lower budgets exhibit lower spread and narrower boxes. To investigate further, I created separate boxplots for budget tier pairs. All budget tiers have outlier values.

For movies with low and mid budgets, I noticed that the median does not align with the center of the box, indicating significant skewness in the revenue data for these tiers. Additionally, a considerable number of outliers are present in the low and mid budget tiers, prompting further investigation. Upon cross-checking some data, I concluded that these outliers represent movies that performed exceptionally well, occurring more by chance than error.

Considering the left-skewed nature of the 'Year' column, I have decided to focus my analysis on movies released in approximately the last 40 years. This choice is motivated by limited and mostly missing data for older movies and the need to adjust revenue and budget figures for inflation to ensure objective comparisons. Upon inspection, I discovered only a small subset of movies released before 1980, which comprise 95% of the initial dataset.

old_movie_count = Movies[Movies['YEAR'] <= 1980].count()

df_adj =Movies[Movies['YEAR'] >= 1980]
Budget and Revenue

In this section, I will conduct a time-series and relationship analysis for the budget and revenue columns. The first graph below illustrates the median revenue and budget over the years. Notably, the revenue line consistently surpasses the budget line throughout history. However, I would like to draw attention to the widening gap between these two measures, particularly in recent years.

df_revsum = df_adj.groupby('YEAR')['REVENUE'].median()
df_revsum.plot()
df_revbudget = df_adj.groupby('YEAR')['BUDGET'].median()
df_revbudget.plot()

plt.title('Median Revenue vs. Median Budget')
plt.ylabel('Million')
plt.xlabel('Year')
plt.legend(["Revenue","Budget"], loc=2)

I was intrigued to identify the movie genre responsible for the substantial revenue growth observed from the year 2000 and present. Therefore, I created a stacked area chart displaying the median revenue grouped by the eight most popular genres and others. The chart reveals that the revenue surge during those years originated from movies in genres such as comedy, drama, thriller, science fiction, horror, and Other.

genres_to_replace = ['Adventure', 'Crime', 'Family', 'Fantasy', 'History',
                     'Music', 'Mystery', 'War', 'Western']
df_replaced = df_adj.copy()
df_replaced['primary_genre'] = df_adj['primary_genre'].replace(genres_to_replace, 'Other')

df_rev_growth = df_replaced[df_replaced['YEAR'] >=2000]
table = pd.pivot_table(df_rev_growth,values='REVENUE', index='YEAR', columns='primary_genre', aggfunc='median')
table.plot(kind='area')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.ylabel('Revenue (mil)')
plt.xlabel('Year')
plt.title('Median Revenue by Movie Genres (2000-2023)')

To delve deeper into this phenomenon, I aimed to explore the factors driving the sudden revenue spike over the past five years. In the line graph below, I plotted the movie count over time, categorized by budget tiers.

Remarkably, the count of high-budget tier movies has exponentially increased in the last five years. Initially, this trend may be attributed to the effects of COVID-19 combined with rising production costs. An online research of this topic revealed that "big studios seek big blockbusters, especially during COVID-19, as they generate more revenue," and "due to COVID, budgets from previous years cannot be directly compared to film production budgets during the pandemic."

low_t = df_adj[df_adj['budget_tier'] == 'Low']
mid_t = df_adj[df_adj['budget_tier'] == 'Mid']
high_t = df_adj[df_adj['budget_tier'] == 'High']
bb_t = df_adj[df_adj['budget_tier'] == 'Blockbuster']

df_low_t = low_t.groupby('YEAR')['budget_tier'].count()
df_low_t.plot()
df_mid_t = mid_t.groupby('YEAR')['BUDGET'].count()
df_mid_t.plot()
df_high_t = high_t.groupby('YEAR')['budget_tier'].count()
df_high_t.plot()
df_bb_t = bb_t.groupby('YEAR')['BUDGET'].count()
df_bb_t.plot()

plt.title('Movie Count by Budget-Tier Over Time')
plt.ylabel('Count')
plt.xlabel('Year')
plt.legend(["Low","Mid","High","Blockbuster"], loc=2)

A bar chart below depicts the budget distribution across different genres of movies. It is evident that genres like Comedy, Drama, and Horror have comparatively lower budgets, reflecting the lower costs involved in creating such movies. In contrast, genres like Animation, Science Fiction, Action, and Romance require higher budgets.

df_genres = df_replaced[df_replaced['YEAR'] >=2000]

df_gmed = df_genres.groupby('primary_genre')['BUDGET'].median().sort_values(ascending=False)
df_gmed.plot(kind='bar')

plt.title('Median Budget by Genre')
plt.ylabel('Million')
plt.xlabel('Genre')

Additionally, it is interesting to examine the relationship between budget and revenue. I aim to determine if an increase in budget correlates with higher movie revenue. The scatterplot clearly demonstrates a moderate positive correlation between a movie's budget and its revenue. This correlation is logical since occasionally, movies may underperform despite having a large budget.

sns.scatterplot(
    x="REVENUE",
    y="BUDGET",
    data=df_adj,
    alpha=0.3)
plt.title('Revenue vs Budget Relationship')

plt.tight_layout()

Income

Another intriguing aspect of the data is the profitability ratio of movies grouped by genres. The first stacked bar chart below depicts the total movie count categorized by genre and stacked by the count of profitability (yes/no) values. Notably, drama, action, comedy, horror, and animation are the genres with the highest movie counts in the last 43 years.

Examining the stacked percentage bar chart, we can observe the profitability ratio for each genre. Although there are no major differences, genres such as animation, comedy, and family exhibit the highest percentages of profitable movies.

grouped = df_adj.groupby(['primary_genre','profitable']).size().unstack()
sorted_df = grouped.sum(axis=1).sort_values(ascending=False)
grouped_sorted = grouped.loc[sorted_df.index]
grouped_sorted.plot(kind='bar', stacked=True)
plt.xlabel("Main Genre")
plt.ylabel("Count of Movies")
plt.title('Total Movie Count by Genre')

cross_tab_prop = pd.crosstab(index=df_adj['primary_genre'],
                             columns=df_adj['profitable'],
                             normalize="index")
cross_tab_prop.plot(kind='bar',
                    stacked=True,
                    figsize=(6, 5))

plt.legend(loc="upper right", ncol=2)
plt.xlabel("Main Genre")
plt.ylabel("Percentage")
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.title('Movie Profitability Proportions by genre')
plt.show()

Relationship Heatmap

By referring to the correlation heatmap, we observe that only the income & budget and revenue & budget relationships demonstrate a moderate correlation. The remaining relationships do not indicate any significant correlation. It is worth noting that the nearly complete positive correlation between income and revenue, is not considered significant due to dataset characteristics.

df_corr = df_adj[['income_mil','BUDGET','SCORE','REVENUE']].corr()
sns.heatmap(df_corr, annot=True).set_title('Correlation Matrix')

Summary and conclusions Movies with higher budgets exhibit wider spreads in terms of revenue, while those with lower budgets have narrower spreads. Outliers can be found across all budget tiers, and movies with lower budgets often show more skewed revenue data, likely due to a some exceptionally successful films.

Over the past four decades, movie budgets have increased by a factor of 2, while movie revenues have significantly increased by a factor of 7. The disparity between budgets and revenues has widened, particularly from 2018 to 2023. Generally revenues have increased across all genres, but the increase can be largely attributed to genres like comedy, drama, thriller, science fiction, and horror.

The count of high-budget movies has notably risen in the past five years. These movies generate larger revenues, partly due to a more diverse revenue distribution. Big studios prioritize high-budget films for greater revenue generation, particularly during the pandemic, contributing to the rise in their count.

Among the movies in the original dataset, around 65% belong to the top five primary genres: Drama, Action, Comedy, Animation, and Horror. These genres have a higher count because they are more cost-effective to produce, resulting in easier accessibility for filmmakers. Consequently, movies with lower budgets tend to have higher counts.

Only 17% of the movies in the dataset have a negative income (budget-revenue). Certain genres like Documentary, Animation, and TV Movie have a slightly higher proportion of profitability, while genres like War, Crime, and Adventure have a lower likelihood of being profitable.

There is a moderate positive correlation between a movie's budget and its income, and a stronger positive correlation between budget and revenue. This relationship is expected, although it is worth noting that movie performance can still vary regardless of the budget.

Movie Recommendation system

Movies.head()

# importing basic library for movie recommendation
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
Movies['index'] = range(0, len(Movies))     # creating new column "index"

Movies = Movies.set_index('index').reset_index() # move the column to begining of the dataset
# select the  reqired column for find some patterns between each movies
selected_features = ['GENRE',"ORIG_LANG"]
print(selected_features)

for feature in selected_features:
    Movies[feature] =Movies[feature].fillna('')

# combining all the 5 selected features

combined_features = Movies['GENRE']+' '+Movies['OVERVIEW']
['GENRE', 'ORIG_LANG']
# initialize the model
vectorizer = TfidfVectorizer()

# fit the data into the  model
feature_vectors = vectorizer.fit_transform(combined_features)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
# make the recommendation based on your movie which you like
# get input of the movie_name

movie_name =input('Enter your favourite movie name : ')

# here we define some default name  -----> otherwise you can get it from user
movie_name = "Iron man"
print('Enter your favourite movie_name : Iron man')

list_of_all_titles = Movies['NAMES'].tolist()       # get all movie name as list

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)  # find closest match --list of movie-- form oru given movie
print(" similiar names  :  ",  find_close_match ,"\n\n")

close_match = find_close_match[0]             # we take first one which is given by cloest movie

index_of_the_movie = Movies[Movies.NAMES == close_match]['index'].values[0]   #  geting the index of the movie

similarity_score = list(enumerate(similarity[index_of_the_movie]))            # generate the similarity score for the given movie

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)     # sort the score to get top 10 more similiarty movies

print('Movies suggested for you : \n')

i = 1

# print the top 10 movies base on similiarity score

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = Movies[Movies.index==index]['NAMES'].values[0]
    if (i<10):
        print(i, '.',title_from_index)
        i+=1

Based on given data we find similiarity and genrate some recommendation

Revenue Prediction Model Building

# importing basic libraries for model building

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
import xgboost as xgb
# done some data processing for model building
Movies= pd.concat([Movies, Movies['GENRE'].str.get_dummies(',')], axis=1)
val1=Movies.iloc[:,14:]
val2=Movies.loc[:,["SCORE","BUDGET"]]

new_data=pd.concat([val1,val2],axis=1)
# to check the co-relation between the "revenue " and other columns

numeric_columns = Movies.select_dtypes(include=['float64', 'int64','int32']).columns

for i in numeric_columns:
    if i != "REVENUE":  # Exclude the target column
        correlation = Movies["REVENUE"].corr(Movies[i])
        print(f"{i}: {correlation}")

numerical=["BUDGET","SCORE","Release_year"]

#  initialize the  independent values in "x"  _________dependent or target in "y"

x = Movies[["BUDGET","SCORE","Release_year"]]
y = Movies["REVENUE"]
# split the x,y train and test data  using "train_test_split"   20% ---> test  :  80% ---> train

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

print(x_train.shape,x_test.shape)

# importing module for data preprocessing

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  StandardScaler,MinMaxScaler

# here we scale the orginal data and normailze the data
column_transformer = ColumnTransformer(
    transformers=[

        ('scaler', MinMaxScaler(), numerical)
    ],
    remainder='passthrough'
)

# Fit and transform the data
featured_xtrain = column_transformer.fit_transform(x_train)

featured_xtest = column_transformer.transform(x_test)
# Initialize the required model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200,ccp_alpha=0.1,criterion="poisson",min_samples_split=60)

model_xgb = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.2, learning_rate = 1.5,
                max_depth = 300, alpha = 2, n_estimators = 10)
# fit the dependent and independent data to the model
model.fit(featured_xtrain,y_train)

# fit the dependent and independent data to the model
model_xgb.fit(featured_xtrain,y_train)

# make prediction by using train dataset
score = model.predict(featured_xtrain)
score_xgb = model_xgb.predict(featured_xtrain)
# make prediction by using test dataset
score_in_test =  model.predict(featured_xtest)
xgb_score_in_test= model_xgb.predict(featured_xtest)
from sklearn import metrics
# use the train data score to find accuracy

model_score= metrics.mean_squared_error(y_train,score)                             # Randomforest regression
model_score_xgb= metrics.mean_squared_error(y_train,score_xgb)                     # XGBOSTER

r2_score= metrics.r2_score(y_train,score)                                           # Randomforest regression
r2_score_xgb=  metrics.r2_score(y_train,score_xgb)                                  # XGBOSTER
# use the test data score to find accuracy

model_score_test= metrics.mean_squared_error(y_test,score_in_test)                    # Randomforest regression
model_score_xgb_test= metrics.mean_squared_error(y_test,xgb_score_in_test)            # XGBOSTER

r2_score_test = metrics.r2_score(y_test,score_in_test)                                 # Randomforest regression
r2_score_xgb_test =  metrics.r2_score(y_test,xgb_score_in_test)                        # XGBOSTER
print("Mean squared error in Random forest: \n","\ntrain data :",model_score,"\ntest data: ",model_score_test)
print("\n\nR2 score in Random forest: \n","\ntrain data :",r2_score,"\ntest data: ",r2_score_test)
print("\n\n")
print("Mean squared error in XGBoost : \n","\ntrain data :",model_score_xgb,"\ntest data: ",model_score_xgb_test)
print("\n\nR2 score in  XGBoost: \n","\ntrain data :",r2_score_xgb,"\ntest data: ",r2_score_xgb_test)

plt.figure(figsize=(20,8))
plt.scatter(y_test,xgb_score_in_test)
plt.scatter(y_train,score_xgb)
plt.legend(["pridict","orginal"])

plt.show()

plt.figure(figsize=(20,9))
plt.scatter(y_test,score_in_test)
plt.scatter(y_train,score)
plt.legend(["pridict","orginal"])

plt.show()
