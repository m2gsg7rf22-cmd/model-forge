# ביקורת רכיב: תצוגת FD ציבורית בלבד

אושר הרכיב fresh_rank_view בלבד, על הבתים החתומים. לא אושרו full fresh protocol/source, RNG, יצירת teacher/נתונים, שאילתות או actual experiment. המשימה הרחבה ושער המודל הפרטי נשארים פתוחים/סגור בהתאמה.

FDView דורש שני commitments מפורשים ולא־ריקים ל־cases.json ול־manifest FD בלבד. סכמה מדויקת מונעת שדות width/seed/type/context size, רשומות context/reference, כפילויות JSON ושמות/קבצים נוספים. קבצי האימות והפלט אינם כוללים parsing של metadata במיקום truth סמוך. כל קובץ נתונים עובר אימות SHA/header/shape/dtype לפני שימוש במערך, ו־svec הקפוא דורש סימטריה מדויקת. מיפוי packing הוא משולש עליון row-major עםsqrt2 מחוץ לאלכסון.

בפרוב עצמאי מילולי נבדקוn3/output2/probes9, ערוץFP32 ו־chunk שני חלקי. חישוב nested-loop ידני תאם בדיוק לכל18 השורות×6 עמודות. כל שלושת המטריצות סומנוreadonly; ששתused hashes הופיעו בדוח verify_unchanged. traps עלopen/stat/iterdir אישרו שלא נעשתה גישה ל־truth הסמוך. אין בפרוב RNG או נתון מדעי אמיתי.

פרובים נוספים אישרו דחייה לפניread_arrays של width/type/seed/context_bytes, boolgeometry, חריגות geometry, null/nonhexSHA, boolsize ו־manifest החושף context bytes. נבדקו גם case לא־ידוע, header shape שגוי, metadata שהשתנה ושינוי same-size בקלט שכבר נצרך לאחר הקריאה האחרונה אך לפניreturn. האחרון נתפס ב־verify_unchanged. אין ממצא חוסם שנותר ברכיב.

נבדקו hashes של ששת המקורות הקפואיםmath/store/rank_selection/IO/glu32/antipodal מול הביקורת הדורבלית הקודמת; הם לא השתנו. טיוטת התכנון החתומה גם לא השתנתה. בדיקות ההורה23/mypy1/Ruff2 מדווחות בנפרד ולא הורצו שוב. תוצאות הפרוב העצמאי והמקור המדויק חתומים ב־JSON.

גבולות האישור: הרכיב מאפשר במכוון1..24 מקרים, n/output1..32 ו־probes1..16 לצורך fixtures. הוא אינו מאמת24מקרים אמיתיים, שלושה רוחבים לכלn,12pairings, n=output או16probes; אלה שערי הבקר העתידי. anonymized IDs לבדם אינם מוכיחים שהשמת הרוחב הוסתרה במקור/בסדר generation. Future generator, commitments ומטא־נתונים נפרדים דורשים ביקורת. מקורloader זה אינו OS sandbox או statistical blinding.

verify_unchanged מאמת את כלmetadata/inventory/size ואת hash הקלטים שנצרכו; הוא אינו מחייב hash לכלקובץ שטרםנצרך. בקרשל24בחירות חייב לצרוך/לחתום את כל התצפיות ולוודא unchanged לפניseal/evaluation. readonly NumPy הוא חוזה שימוש רגיל ואינו גבול ביטחון כנגד קוד זדוני שמשנה internals. לא נבדקה כאן תקרתTREE/זיכרון של פרוטוקול מלא; schemas bounded אינם resource receipt.

פרוטוקול variable-width עדיין דורש source/controller/generator/evaluator חדשים, secret commitments, all24seal-beforetruth, sensitivity/full/zero allcase endpoint, inventory/resourcepreflight/globalmarker, sourcefreeze ואישור עצמאי מלא לפני הנתונים. אין ביצוע רשום ואין טענת rank/recovery/functional improvement או שחרור שער פרטי.
