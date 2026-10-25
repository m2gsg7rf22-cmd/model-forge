# ביקורת תכנון — אבחון גאומטריית normals לקריאה בלבד

התכנון מתאים לבדיקת ישימות הדומיין על כיול ציבורי שמור, ללא אופטימיזציה. עדיין נדרשת סקירת מקור/פרוטוקול/בקר לפני כל קריאת נתוני ניסוי. נקראה התוכנית docs/superpowers/plans/2026-10-25-guard-normal-geometry-diagnostic.md.

הפרדה הכרחית בסיכומים: 64 יחידות מתוכננות; mapping תקין או כשל מפורש לכל אחת; ורק אחר כך geometry תקינה/מחוץ לדומיין/שגיאה נומרית. אין להציג אחוז pass מתוך יחידות שהצליחו בלבד כאילו כלל הכשלים הושמטו. כשל logical ties/zeroM/reference/normal תחת המדיניות הקפואה הוא תוצאה שמורה, לא היתר לקצץ שורות או לבחור יחידה אחרת.

A ללא שורות אחרי השמטת computed-zero normals דורשת תוצאה מפורשת: חרוט בלתי מוגבל באילוצים, mask0, spectrum ריק, ratio לא ישים. logical records נשמרים. אין להשמיט את היחידה או לסווג את המקרה כ־zero-M, שכבר נדחה בשלב mapping. לכל A תקין enumerate בדיוק2^m masks; לכל mask רשומים active indices, row count, singular values של normalizedrows, ratio/סיבה, ואם נדרש exact rank על ORIGINALA בלבד. יותר שורות מממד אינן independent; תלות מדויקת ו־near independence מחוץ לדומיין הן קטגוריות שונות. SVD/error או אי־ייצוג נומרי אינם אישור דומיין. בדיקת כל subset כאן היא תיאור ולא קריאה למחשב QP.

הפרימיטיב אינו פותר direction, KKT או Gauss–Newton ואינו קורא cone_direction/np.linalg.solve. העובדה שהמודול הקפוא כולל פונקציות כאלה אינה בעיה אם נקרא ממנו רק exact_row_rank ומקורו מחויב בחתימה. ה־metric mapping משתמש בשגיאת המועמד המלא ובשורות guard בלבד; רשומות raw normals/metrics, ניבוי ו־physicalerror נדרשים לשחזור עצמאי ללא solve.

ה־fullfitreport3MB נשאר מחויב בחתימה כנתונים גולמיים בלבד ולא דרך parser2MB. compact metadata1976bytes חייב להיבדק מול חתימתו, full_report_sha/flag, candidate/currentbaseline ו־manifests. כל seed/scale/rowcount/channel של calibration המקורי קבוע; guard בשימוש חוזר הוא כיול אדפטיבי, לא בדיקה חדשה. אין צורך ואסור לקרוא public-validation מהשחזור או מתוצאות קודמות.

בדיקת התעבורה חייבת להפעיל את קוד serialization האמיתי על64×256רשומות תתי־קבוצות, logical metadata ומטריצות, לשמור fullJSON/NPZ חיצוניים ולהחזיר receipt דרך מנוע מוגבל2KiB. בדיקת המערך4.8MB מהשלב הקודם לבדה אינה מוכיחה שה־schema החדש יישאר קומפקטי. יש לחייב hashes לכל evidence file; summary גדול יכול להישאר בקובץ בעוד receipt_stdout קטן קבוע. כשל transport אינו מצדיק חזרת data diagnostic אוטומטית או parser relaxation.

180שניות/512MiB בעץ עובד וב־BLAS יחיד נשארים תקציב וניטור מדגמי, לא הבטחת שיא זיכרון רציף. Fraction rank על עד8×32ערכיFP64 הוא bounded arithmetic אך אינו תעודת זמן קבועה; timeout/failure נשמרים. יש להקפיא geometry1e-5 ו־nearTie1e-12 לפני ההרצה. אין לשנותם בעקבות אבחון, להפעיל אופטימייזר או להשתמש ב־941032שנצרך. אבחון זה אינו רושם/צורך941033. הכשלים הקודמים והשער הפרטי הסגור נשמרים.
