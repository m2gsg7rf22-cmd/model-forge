סקירת פרוטוקול ראשונית — טרם חתימה סופית

הבקר מפריד בין אימות קוד/רישום לבין קלטים מדעיים. preflight נדרש לפני הפעלת worker, ולא לפניו נקרא או מגובב קלט מדעי. deadline משותף מופחת לפני worker ונבדק שוב אחרי אימות התוצרים. גם קבלת worker חייבת process_and_children. אין evaluator ואין קריאה או רישום של 941033. root בלעדי וכשל סופי חוסמים המשך; קבלה מוצלחת אינה טענת איכות.

verify_source קושר אישור actual_fitting_protocol_approved, מלאי 14 קובצי תלות, רישום מוקדם, גבולות זמן/זיכרון/פלט/BLAS ומצב clean discovery-lab. המקור נבדק שוב אחרי העבודה ונדרש HEAD זהה. input adapter בודק שישה גיבובים קבועים, שרשרת compact/full-report/raw/original manifests, בדיוק את שני זרעי train/guard וערוצי FP32; full report נשאר hash-only. הקוד אינו מייבא מורה או evaluator. אין מנגנון שמבדיל physicalY מ־weightedY לפי ערכים בלבד; כאן שרשרת הגיבובים היא הבסיס להסכמה זו.

פרוב בקר עצמאי חדש מחליף engine ו־source gate בלבד בקבלות סינתטיות; אינו מפעיל subprocess או קורא קלטים אמיתיים. הצלחה עם שינוי נתנה eligibility; הצלחה ללא שינוי לא נתנה. leftover candidate, תוספת קובץ למלאי, failed root וקבלת worker ללא ניטור ילדים — כולם נעצרו בכשל ללא eligibility. נשמרו controller_probe.py ו־controller-probe-results.json.

ממצא פתוח לפני חתימה: worker שומר inputs_before/after רק לאחר הצלחה. נדרשת שמירת before מיד אחרי hashgate ולפני פענוח, ובדיקת after גם ב־finally על כשל מטופל. כשל חיצוני/kill עשוי למנוע finally; יש לציין מגבלה זו במפורש ולא לטעון שימור מדוד ללא קבלה.

אין אישור בפסקה זו להתאמה בפועל; חתימת המקור המלא תינתן רק לאחר סגירת הממצא, גיבוב בתים סופיים ורישום המקור. polling אינו הבטחת שיא רציף; RAM של הבקר אינו מכוסה אוטומטית ב־worker RSS. אין צורך לחזור על בקרת transport גדולה שכבר בוצעה.
