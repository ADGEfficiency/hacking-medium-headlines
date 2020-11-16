## Log

Used `https://towardsdatascience.com/`
- difficult parsing (classes change)

Used 'https://towardsdatascience.com/archive'
- easier :)


## Outcomes / goals

CLI to predict claps from headline (use from a shell)

Interpret features using LIME

tech
- all in on skearn pipes
- use naive bayes

cookie
- should I just have data/raw, data/processed (no need for interim)


## Open questions

Accuracy by class

Should I rework the binner to have edges I like?

Create features is doing too much

Should I bother with subtitle
- often cut off (see 2014 archive)
- if it's cut off, won't matter for engagement ...

Is parsing the high quality titles a good idea?

Claps rounded to the nearest 1,000 for large claps

Is date important?

Should the claps be normalized by TDS popularity?

Features
- use of numbers (0 to 9, 10+)
- me, I, my (self reference detector)
- LENGTH! (log length)

Should I convert `6` to `six` (or vice versa)?

Classification or regression?
- classification using naive bayes

Need to check for duplicates!!!

CHANGE DIRS TO PATHLIB!!!
