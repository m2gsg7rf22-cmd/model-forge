ביקורת מקור עצמאית — הערכה ציבורית זוגית חד־פעמית

אישור מותנה בקיבוע הבתים המדויקים שב־source-review.json ב־commit נקי וב־preflight tree לפני נתונים. זהו אישור לניסיון יחיד של 2048 וקטורי reference חדשים ב־941033 עבור המועמד והבסיס הקפואים בלבד. אין אישור התאמה, retry או שינוי קריטריון. אין הערכה אמיתית שבוצעה כחלק מביקורת זו.

נבדקו evaluator, controller, רישום, מסמכי חוזה ותוכנית ובדיקות. שמונה תלויות runtime ורישום הגיבובים קושרים את שני המודלים, תוצאת ההתאמה, worker/step/manifest וביקורת התוצאה העצמאית. input adapter דורש terminal fit success, tree receipts, updated candidate, fixed C ומלאי שלם. הסמן הקבוע נוצר בלעדית בשורש ההתאמה, מחוץ ל־step inventory, לפני factory ולפני יצירת נקודות. קובץ חלקי אינו נמחק ואינו מאפשר retry.

הממצאים שנמצאו במהלך הסקירה תוקנו: norm underflow אינו מתפרש כשגיאה0; שינוי input בכל אחד משלושת callbacks מכשיל; reference ו־prediction buffers מנותקים כדי שמחזור זיכרון רגיל לא ישנה תוצאות קודמות; progress קריאתו מוגנת ואינו מסתיר את החריגה המקורית או מונע בדיקת after; reservation_creation_started/null מופרד מ־metadata_persisted. כשל כתיבה או הריגה עדיין עלולים למנוע מסמך כשל, ואין טענת fsync או עמידות עסקית בקריסה.

פרובים עצמאיים השתמשו בזרעים חדשים 25,27,28,29,30,33,34,35 בלבד, במערכים זעירים וב־callbacks סינתטיים. לא נוצרו נקודות seed941033, לא הופעל public reference factory, לא נקרא consumedvalidation ולא נוצר marker941033. generic marker אימת חסימת ניסיון חוזר גם לאחר serialization חלקי. פרוב batch אימת buffers ממוחזרים מנותקים, הצלחה בספירות [8,8,8,8], כשל query שני בספירות [8,4,2,2], ושמירת reference לפני כשל חיזוי.

פרוב controller החליף source gate ו־engine בקבלות סינתטיות בלבד; marker metadata הוחזק בזיכרון בלי קובץ marker אמיתי. נבדקו success איכות, qualityFAIL שהינו completed תקין, וכשלי worker scope, input mutation, counters ו־marker tamper. כל הכשלים נעצרו ללא quality claim. source gate עצמו נסקר סטטית: exact8dependencies/prereg/approval/clean discovery-lab/limits, עם בדיקה חוזרת אחרי העבודה. preflight קודם לכל scientific hash/read, ונדרשת גם קבלת tree של worker.

בפועל יש להריץ רק אחרי clean exact commit, עם output חדש ובלעדי, deadline משותף180s,512MiBדגום/BLAS1/2KiBפלט. reserved/attempted/returned/persisted הם מושגים שונים; attempted הוא intent, ו־missing progress אינו0. נשמרים raw inputs/reference/predictions לכל group לפני המשך, ומדווחים שני המדדים והפרשים/החמרות כולם. הקריטריון נשאר כל4relative<=0.001. completed אינו בהכרח qualityPASS.

לאחר הריצה נדרשת ביקורת saved-output נפרדת, ללא queries או fit נוסף. RSS דגום אינו הוכחת שיא רציף או זיכרון הבקר; זמן הקבלה אינו עסקת זמן קשיחה מול עצירות חיצוניות. fixture ציבורי אחד אינו sealed hidden-teacher, שקילות לכל קלט, שחזור Transformer או יכולת frontier. כל הכשלים הקודמים נשארים; השער למודל הפרטי סגור.
