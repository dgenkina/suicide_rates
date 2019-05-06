# suicide_rates
The intent of this project is to study the dependence of suicide rates on cultural norms and values, teasing out contributing factors beyond economics. Insights from this project could be potentially useful to governments and world aid organizations, guiding their resource allocation decisions for maximal impact, as well as individual psychotherapists, informing them which cultural trends are to be encouraged and which are best avoided. 

To tackle this question, I downloaded two datasets: a Kaggle dataset containing suicide rates by country from 1985 to 2016 (https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016), and results of an international research study by the Globe Project measuring societal beliefs on a number of topic by country (https://globeproject.com/study_2004_2007#data). I combined these two datasets by country, throwing out countries that occurred in one dataset and not the other. 

The first graphic looks at the average suicide rates per 100,000 population between 1985 and 2016 (top panel) and the slope of the suicide rate in occurrences per 100,000 population per year (top panel) as a function of cultural region. The graphic shows that suicides rate per cultural region vary by almost a factor of four, with the highest rate occurring in Confucian Asia and the lowest in the Middle East.  The slopes as a function of time also vary by a factor of three, with the steepest increase (alarmingly) also in Confucian Asia, while the steepest decrease is in Nordic Europe.

To tease out which of the cultural metrics measured by the Globe Project had the highest impact (if at all) on suicide rates, I ran logistic regression. The results of this test are shown in the second graphic. First, the R^2 value of the regression analysis was 0.7, meaning the parameters measured could explain approximately 70% of the variation in suicide rates. Notably, this number did not improve when more conventional metrics like per capita gdp and the human development index (HDI) were included in the fit. This suggests that the cultural factors have significant explanatory power for variation in suicide rates. 

Secondly, we observe that the strongest predictor of suicide rates is measured 'power distance practices'. The Globe Projects describes this metric as 'the extent to which the community accepts and endorses authority, power differences, and status privileges'. It is not surprising that this correlates positively with higher suicide rates as deferring to authority often comes at the cost of the individual. It is more surprising how strong this correlation is compared to others. One might expect, for example, that 'humane orientation', described as 'the degree to which a collective encourages and rewards (and should encourage and reward) individuals for being fair, altruistic, generous, caring, and kind to others', might be a strong negative predictor of suicide rates, however it appears to be less important than power distance practices by a factor of 4. 

The most shocking result is perhaps the positive correlation between suicide rates and 'gender egalitarianism practices', described as 'the degree to which a collective minimizes (and should minimize) gender inequality'. Since male suicide rates are generally higher than female, one might suspect that making a society more egalitarian pushes women into the same situations as men, leading to an increase in female suicide rates. However, this trend is consistent across both genders: higher levels of gender egalitarianism suggest both higher suicide rates by women and men. 
