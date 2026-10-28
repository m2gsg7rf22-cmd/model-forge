# אודיט עצמאי — variable-width gate development v1

האודיט עבר: הראיות השמורות עקביות עם המסלול והדוחות הרשומים. הפלט המדעי מוגבל: רק מקרה אחד הגיע למספר המועמדים r/2, וחמישה סיימו את ארבעת סבבי ה־rescue ללא הגעה ליעד. לא נבדקו זהות gates או כיסוי.

מקור הניסוי `43cf7a6cc04b88d1cf1b3071fcf1fe437dd38eae`. אומתו28 חתימות מול commit המקור והמצב הנוכחי, ancestry, ‏14 runtime pins, config/review/marker, ‏54 ארכיוני public ו־24 estimator של ההורה, ‏5,064 trace ו־20 transport. נבדקו כל12 בחירות המקור וה־seal לפני חילוץ ששת בסיסי stable-p17. אין קריאת truth או קלטי מורה.

| job | rank | יעד r/2 | מספר בסיס | מספר סופי | חסר ליעד | rescue rounds | מצב | loss מרבי במקור |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| job00 | 192 | 96 | 81 | 93 | 3 | 4 | rounds_exhausted | 7.71668214545e-09 |
| job01 | 192 | 96 | 1 | 2 | 94 | 4 | rounds_exhausted | 9.47705317086e-09 |
| job02 | 96 | 48 | 47 | 47 | 1 | 4 | rounds_exhausted | 4.60735330838e-09 |
| job03 | 96 | 48 | 18 | 18 | 30 | 4 | rounds_exhausted | 9.83711408294e-09 |
| job04 | 32 | 16 | 16 | 16 | 0 | 0 | candidate_target | 1.56692984913e-09 |
| job05 | 32 | 16 | 13 | 13 | 3 | 4 | rounds_exhausted | 8.90193329077e-09 |

כל סדרי המיזוג, ההחלטות accepted/duplicate/loss, nearest indices והספירות חושבו מחדש מהכיוונים וההפסדים השמורים בספים המקוריים בלבד. כל מסלול הכיוונים הסופי תואם בדיוק לרצף המיזוג. נבדקו units, original-basis losses, pretraces, כל intents/chunks והעתקי aggregate, מצבי RNG המתועדים, count/status ודוחות המקרים.

גאומטריית tangent/component/residual חושבה מחדש באמצעות SVD מהכיוונים והבסיסים השמורים, ללא אופטימיזציה. הממדים, הסינגולריים הקטנים וה־residual projectors תואמים. משוואות eigen, אורתוגונליות, indices ובחירת starts נבדקו מתוך eigenpairs שמורים, בלי eigensolve. כל per-view losses נבדקו מהבסיסים הציבוריים.

הפרש מרבי בהפסדי המיזוג/הכיוונים הסופיים, residual projector, tangent singular values ו־view residuals: אפס. שארית משוואת eigen מרבית1.0130785099704553e-15; פער chunk-loss לעומת aggregate עד1.0545560193316039e-19, בתחום הסבילות שנקבעה מראש.

## חלוקת החלטות המיזוג

אלה קטגוריות תיאוריות בלבד, ללא זיהוי אמת או שינוי סף. loss rejection הוא loss>1e-8; duplicate rejection חל על מועמד שעבר loss אך abs cosine>=.9995.

| job | שלב | קבלה | כפילות | דחיית loss |
|---|---|---:|---:|---:|
| job00 | initial | 81 | 161 | 14 |
| job00 | rescue0 | 12 | 405 | 95 |
| job00 | rescue1 | 0 | 6 | 506 |
| job00 | rescue2 | 0 | 5 | 507 |
| job00 | rescue3 | 0 | 4 | 508 |
| job01 | initial | 1 | 2 | 253 |
| job01 | rescue0 | 1 | 4 | 507 |
| job01 | rescue1 | 0 | 0 | 512 |
| job01 | rescue2 | 0 | 0 | 512 |
| job01 | rescue3 | 0 | 0 | 512 |
| job02 | initial | 47 | 203 | 6 |
| job02 | rescue0 | 0 | 89 | 423 |
| job02 | rescue1 | 0 | 94 | 418 |
| job02 | rescue2 | 0 | 87 | 425 |
| job02 | rescue3 | 0 | 85 | 427 |
| job03 | initial | 18 | 80 | 158 |
| job03 | rescue0 | 0 | 1 | 511 |
| job03 | rescue1 | 0 | 0 | 512 |
| job03 | rescue2 | 0 | 1 | 511 |
| job03 | rescue3 | 0 | 0 | 512 |
| job04 | initial | 16 | 240 | 0 |
| job05 | initial | 13 | 198 | 45 |
| job05 | rescue0 | 0 | 1 | 511 |
| job05 | rescue1 | 0 | 0 | 512 |
| job05 | rescue2 | 0 | 1 | 511 |
| job05 | rescue3 | 0 | 0 | 512 |

סך הכול 11,776 מועמדים נבדקו במיזוג: 189 קבלות, 1667 דחיות כפילות ו־9920 דחיות loss. הסכום אינו recall, precision או coverage. אין להסיק מכך שחרור סף או שינוי אלגוריתם ללא פרוטוקול נפרד.

## משאבים ושלמות

האודיט רץ בניסיון מספרי אחד לאחר preflight TREE שעבר: 16,941,056 בתים/3 דגימות. ה־numeric audit: process_and_children, ‏154,402,816 בתים מדוגמים/68 דגימות, במסגרת180 שניות/512MiB/BLAS1/2KiB. זמן launcher כולל preflight וקבלות 4.511105708 שניות.

הריצה המקורית: TREE ‏392,577,024 בתים/1499 דגימות; זמן פנימי לפני כתיבת הדוח הסופי 95.066099916 שניות. retained bytes לאחר כתיבת outer: ‏135,203,053; trace כולל manifest ‏126,366,332, transport כולל manifest ‏8,767,097. כל החסמים, קבלות TREE, exact layout וחתימות לפני/אחרי עברו.

## מגבלות

- לא נוצרו generators מהזרעים הרשומים ולא שוחזרו RNG draws. נבדקו תחומי SHA256, seed-map/marker/config ומבנה PCG64 states והתאמתם לדוחות; seed→state ודגימות normal נשענים על המקור החתום ולא על replay עצמאי.
- לא הורצו Adam, refine, optimizer או eigensolver באודיט. נבדקו endpoints, מונוטוניות עידון והפסדים מתוך הכיוונים השמורים; אין טענה לשחזור עצמאי של כל1600 צעדי Adam או12/24 צעדי העידון. SVD שימש רק לבדיקת גאומטריה שמורה.
- משוואות eigen, אורתוגונליות, שחזור מטריצת mixture ובחירת שני הערכים המוחלטים הקיצוניים אומתו על eigenpairs השמורים, ללא חיפוש או fitting מחדש.
- לא נקראו teacher factors, truth/context, heldout שנצרך או נכסים פרטיים; לא בוצעו forward, שאילתות או יצירת מורה. האודיט מוגבל למקורות חתומים, public/estimator של ההורה ו־traces של הניסוי.
- מספר r/2 הוא היפותזת משפחה בלבד. רק job04 הגיע למספר היעד; גם תוצאה זו אינה מאמתת זהות gates, כיסוי, מגניטודות, couplings, פונקציה או משקולות.
- חמש עצירות אחרי ארבעה סבבים אינן הוכחת התכנסות, גבול אופטימלי או סיבת כשל. חלוקת loss/duplicate/accepted היא תיאורית בספים הקפואים בלבד, ללא מדד recall או שינוי סף.
- נתוני הפיתוח ציבוריים ונצרכו. ערוצים וקידומות תלויים, seeds בין jobs עצמאיים; אין להסיק השוואת precision סיבתית או הכללה על מורים חדשים. residuals בתצוגות השונות אינם heldout.
- RSS הוא שיא דגימות process_and_children, לא חסם שיא רציף או הקצאת OS. משאבי הניסוי מתייחסים לכל הבקר, לא למדידה לכל job. הזמן המקורי נרשם לפני כתיבת הדוח הסופי.
- לא בוצע ניסיון חוזר של הניסוי או של ה־numeric audit. שער המודל הפרטי נשאר סגור, ואין כאן תיקון לכישלונות הקודמים או הוכחת שחזור אוניברסלי.

SHA256 דוח המקור: `00aee710089c6225cbdc80ffcece2a23ceaa56197d4838331d147ffabdc51ad2`.
SHA256 תוצאות האודיט: `18c7f2794f0b0a7e093c45fcce20d5dbfb4a0483fe4d0fee02a29f747c8e9158`.

רשימת העתקת הקבצים הקטנים והחתימות ב־copy-manifest.json; המערכים הגולמיים נשארים מחוץ ל־Git.
