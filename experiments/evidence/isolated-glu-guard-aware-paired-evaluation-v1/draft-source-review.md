סקירת מקור הערכה ראשונית — שלושה ממצאים לתיקון לפני חתימה

נבדקו guard_aware_evaluate.py והזרימה הסטטית של validate_fit/build/evaluate_batches. אין אישור להרצת הערכה. הקישור לשבעה קלטים קפואים, fit root מדויק, controller/tree receipts, ביקורת התאמה, מלאי step ו־candidate/base נראה מתאים להפרדת ההערכה מההתאמה. סמן קבוע נוצר לפני factory ולפני יצירת נקודות. הקריטריון והפרשי relative/maxabs נשמרים. לא יובא optimizer על ידי evaluator.

ממצאים:

1. ב־finally של build, קריאת progress.json אינה מוגנת. כשל פענוח/IO/גודל יעלים חריגה מקורית, ידלג על inputs_after וימנע worker-report סופי. יש לשמר progress_check_error נפרד, counters unknown אם אין פרוגרס בר־אימות, ולהמשיך בדיקת קלטים וכתיבת כשל. אין להסיק 0 מעדר קבלה.
2. metrics מקבל norm(delta)=0 עקב underflow גם כאשר delta אינו אפס. פרוב עצמאי [1,1e-200] מול [1,0] החזיר relative=0 למרות maxabs=1e-200. נדרש כשל מפורש על nonzero delta עם norm0 ועל יחס0 מול נורמה חיובית. לא להכריז equality בטעות.
3. callbacks מקבלים x שנשמר קודם ואז משמש כזיכרון חי ל־paired output ולניבויים. פרוב query סינתטי שמשנה x ל־7 סיים completed אף שמערכי group-inputs ו־paired-validation אינם זהים. יש להגן/להעתיק ולוודא זהות bytes אחרי כל callback; reference שחזר צריך להיות מנותק מ־x, ובדיקת שינוי של reference אחרי predictors נדרשת כשיש alias. כשל mutation צריך להפסיק ללא query נוספת.

יש להבהיר גם reservation-write failure: סמן קבוע שנוצר אפילו אם כתיבת תוכנו נכשלה עדיין אוסר retry. report צריך להבחין reserved/unknown/לא נוצר, ולא לטעון 0 בוודאות אם פעולת יצירת הסמן התחילה ונכשלה אחרי יצירה. marker קיים של ניסיון קודם אינו שייך לניסיון החדש. מקור ופרוטוקול הבקר עדיין ממתינים לביקורת.

הפרובים ב־draft_probe.py וב־draft-probe-results.json משתמשים ב־seed25 חדש בלבד, בשתי שורות ובממד2. אין public_weights/factory, יצירת actual941033 observations, query אמיתי או קריאת consumedvalidation. אין צורך לחזור על פרובי fitting/transport הקודמים.

עדכון תיקונים: הפרובים העצמאיים fixed_probe.py אימתו ש־underflow ושלוש נקודות שינוי input (כולל עקיפת readonly באמצעות setflags) כעת נכשלים כראוי. ה־reference מנותק, ו־finally שומר שגיאת progress בנפרד. נמצא פער נוסף שטרם נסגר בעת כתיבה זו: predictor שממחזר output buffer בין קבוצות מאפשר completed בעוד paired-validation prediction שונה מקובץ group prediction. נדרשת העתקת base מיד לפני student והעתקת pred מיד בחזרתו. שוחזר ב־seed28 בלבד.

marker_probe.py בזרעים גנריים 29/30 אימת exclusive marker גם אחרי כשל serialization. marker חלקי נשאר וחוסם retry; קיום כזה אינו הוכחת תוכן metadata מלא. לא נוצר marker941033 ולא הופעל factory אמיתי. אין חתימת אישור מלאה עד סקירת הבקר והרישום וסגירת פער ה־buffer.
