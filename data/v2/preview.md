# v2 data preview (training split only)

## UMWP: answerable problem and its unanswerable twin

### umwp-pair-0129 (SVAMP; twin category: Key information missing)

**answerable** (38 words)

> Tiffany was collecting cans for recycling. On monday she had 3 bags of cans. The next day she found 7 more bags of cans and 44 bags of bottles. How many bags of cans did she have altogether?

**unanswerable** (42 words)

> Tiffany was collecting cans for recycling for 3 days. On monday she had 3 bags of cans. The next day she found 7 more bags of cans and 44 bags of bottles. How many bags of cans would she have after Wednesday?

### umwp-pair-0526 (MultiArith; twin category: Key information missing)

**answerable** (35 words)

> Faye was playing a video game where she scores 7 points for each treasure she finds. If she found 2 treasures on the first level and 6 on the second, what would her score be?

**unanswerable** (40 words)

> Faye was playing a video game where she scores 7 points for each treasure she finds. If she found 2 treasures on the first level and 6 on the second and serveral on the third, what would her score be?

### umwp-pair-0667 (MultiArith; twin category: Ambiguous key information)

**answerable** (36 words)

> At the fair there were 31 people in line for the bumper cars. If 25 of them got tired of waiting and left and 25 more got in line, how many people would be in line?

**unanswerable** (36 words)

> At the fair there were several people in line for the bumper cars. If 25 of them got tired of waiting and left and 25 more got in line, how many people would be in line?

### umwp-pair-0984 (GSM8K; twin category: Ambiguous key information)

**answerable** (59 words)

> Selena got a tip today that amounted to $99. She pampered herself by eating at a 5-star hotel. She indulged herself with 2 steak meals that cost $24 each plate. She also ordered 2 types of burgers which cost $3.5 each, and 3 cups of ice cream which cost $2 each. How much money will Selena be left with?

**unanswerable** (59 words)

> Selena got a tip today that amounted to $99. She pampered herself by eating at a 5-star hotel. She indulged herself with 2 steak meals that cost $24 each plate. She also ordered 2 types of burgers which cost $3.5 each, and several cups of ice cream which cost $2 each. How much money will Selena be left with?

### umwp-pair-1062 (GSM8K; twin category: Ambiguous key information)

**answerable** (35 words)

> Darcy washes and drys 20 shirts and 8 pairs of shorts. If he folds 12 of the shirts and 5 of the shorts, how many more remaining pieces of clothing does Darcy have to fold?

**unanswerable** (35 words)

> Darcy washes and drys 20 shirts and 8 pairs of shorts. If he folds some of the shirts and 5 of the shorts, how many more remaining pieces of clothing does Darcy have to fold?

### umwp-pair-1176 (GSM8K; twin category: Ambiguous key information)

**answerable** (28 words)

> James writes from 1 PM to 4 PM every night. He can write 5 pages per hour. How many weeks will it take to finish his 735-page book?

**unanswerable** (30 words)

> James writes from 1 PM to 4 PM every night. He can write more than 5 pages per hour. How many weeks will it take to finish his 735-page book?

## SQuAD 2.0: one passage, one answerable and one unanswerable question

### squad2-passage-19620cc84022 (Scottish_Parliament)

> Much of the work of the Scottish Parliament is done in committee. The role of committees is stronger in the Scottish Parliament than in other parliamentary systems, partly as a means of strengthening the role of backbenchers in their scrutiny of the government and partly to compensate for the fact that there is no revising chamber. The principal role of committees in the Scottish Parliament is to take evidence from witnesses, conduct inquiries and scrutinise legislation. Committee meetings take place on Tuesday, Wednesday and Thursday morning when Parliament is sitting. Committees can also meet at other locations throughout Scotland.

- answerable: *Taking evidence from witnesses is one of committees' what?* (answer: principal role)
- unanswerable: *Very little of the work done by Scottish Parliament is done where?* (answer: none)

### squad2-passage-a0db929a48c8 (Civil_disobedience)

> LeGrande writes that "the formulation of a single all-encompassing definition of the term is extremely difficult, if not impossible. In reviewing the voluminous literature on the subject, the student of civil disobedience rapidly finds himself surrounded by a maze of semantical problems and grammatical niceties. Like Alice in Wonderland, he often finds that specific terminology has no more (or no less) meaning than the individual orator intends it to have." He encourages a distinction between lawful protest demonstration, nonviolent civil disobedience, and violent civil disobedience.

- answerable: *How much literature has been written regarding civil disobedience?* (answer: voluminous; voluminous literature)
- unanswerable: *Who wrote that its difficult to formulate a definition of grammatical niceties?* (answer: none)

### squad2-passage-b6f799a7fbe9 (Harvard_University)

> Harvard's academic programs operate on a semester calendar beginning in early September and ending in mid-May. Undergraduates typically take four half-courses per term and must maintain a four-course rate average to be considered full-time. In many concentrations, students can elect to pursue a basic program or an honors-eligible program requiring a senior thesis and/or advanced course work. Students graduating in the top 4–5% of the class are awarded degrees summa cum laude, students in the next 15% of the class are awarded magna cum laude, and the next 30% of the class are awarded cum laude. Harvard has chapters of academic honor societies such as Phi Beta Kappa and various committees and departments also award several hundred named prizes annually. Harvard, along with other universities, has been accused of grade inflation, although there is evidence that the quality of the student body and its motivation have also increased. Harvard College reduced the number of students who receive Latin honors from 90% in 2004 to 60% in 2005. Moreover, the honors of "John Harvard Scholar" and "Harvard College Scholar" will now be given only to the top 5 percent and the next 5 percent of each class.

- answerable: *What are the top 4-5% graduating students honored with?* (answer: summa cum laude)
- unanswerable: *What percentage of the student body is accepted to the Phi Beta Kappa honor society each year?* (answer: none)
