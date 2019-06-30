Description of the dataset.
    For the review-driven multi-label music style classification  task, we build a new dataset which contains over 7,172 samples.
    We define an album as a data sample in the data set.On average, each sample contains 2.2 styles and 40 reviews, each review has 12.4 words.Each sample includes a music title,
    a set of human annotated styles, and associated reviews.

The music styles include:
    Alternative Music, Britpop, Classical Music, Country Music, Dark Wave, Electronic Music,Folk Music,Heavy Metal Music,Hip-Hop, Independent Music, Jazz, J-Pop,
    New-Age Music, OST, Piano Music, Pop, Post-Punk,Post-Rock,Punk, R&B, Rock,and Soul Music.

An example of a sample in the proposed task is:
    Music title: Mozart: The Great Piano Concertos, Vol.1
    Styles: Classical Music, Piano Music
    Reviews:
        (1) I've been listening to classical music all the time.
        (2) Mozart is always good. There is a reason he is ranked in the top 3 of lists of greatest classical composers.
        (3) The sound of piano brings me peace and relaxation.
        (4) This volume of Mozart concertos is superb.

Since the data set is collected from a popular chinese music review website, the reviews are written by chinese, and the reviews in the above example are translated from original reviews.


The name, a set of music styles and reviews of each album are written to a json file, the file is named by the album's name. and encoding is 'utf-8'.

Keys and values in json dict of each file:
    name: the music title.
    url: the link to the page of the album in the website.
    rates: The average score users give to the album.
    tags: a set of music styles.
    all_short_comments: the value is a list of dicts, including all review items of a piece of music.
        The key and values of the dict in a review item is:
            name:  the user's name.
            rate: the score user give to the album.
            vote: how many users think the user's review makes sense.
            comment: user review.

Since there are rate and vote of each review, the proposed  music review dataset can also be used for sentiment analysis or review rankings.








