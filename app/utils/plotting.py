from plotly.graph_objs import Bar

def return_figure(df):
    """
    Function to :
        Creates plotly visualizations

    Args:
        df (pandas dataframe): dataframe loaded from the database file

    Returns:
        graphs (list): list containing the 2 dict for 2 plotly visualizations

    """
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_count = df.drop(['id', 'message', 'original', 'genre'], axis = 1, inplace = False).sum()
    category_count = category_count.sort_values(ascending = False)
    category_name = category_count.index

    # create visuals
    graphs = [

        {
            "data": [
                {
                    "type": "pie",
                    "name": "Genre",
                    "domain": {
                        "x": genre_counts,
                        "y": genre_names
                    },
                    "textinfo": "label+value+percent",
                    "hoverinfo": "all",
                    "labels": genre_names,
                    "values": genre_counts
                }
            ],

            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },

        {
            'data': [
                Bar(
                    x = category_name,
                    y = category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Categories',

                'yaxis': {
                    'title': "Count",
                    'automargin': True
                },

                'xaxis': {
                    'title': "Category",
                    'tickangle': 35,
                    'automargin': True
                }
            }
        },

    ]

    return graphs