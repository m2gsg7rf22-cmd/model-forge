ביקורת עיצוב — הערכה ציבורית זוגית 941033

סטטוס: תכנון בלבד. לא הופעל reference factory, לא נקראו נתוני הערכות קודמות ולא בוצעה שאילתה. נסקר קוד guarded_gate_step_evaluate.py הקודם כמבנה בלבד. הרישום החדש והמקור עדיין מחייבים ביקורת וחתימה נפרדות.

ההפרדה בין התאמה להערכה נכונה. יש לקבע מראש מועמד 7f7d69552fe3ad8c27ad17b5f8326038fd7d26a2171ad84d8660d07560aa0a06 ובסיס 2f207b106c8dae1f31348c1b46b4ea4bbd08d1f667a5ddb78fa1c86a8002378b, לאחר הוכחת terminal success של ההתאמה, מקור ומלאי מלא וקבלת tree monitoring. יש לקשור ברישום את גיבוב ביקורת התוצאה העצמאית, לא רק להסתמך על approved מתוך JSON חליפי.

סמן 941033 צריך להיות open('x') קבוע בשורש ההתאמה המקורי לפני factory וטרם שאילתה. הוא אינו נמחק גם אם factory, prediction, כתיבה או ניטור נכשלו. שורש הפלט החדש אף הוא בלעדי. יש להחריג רק את שם הסמן המדויק מבדיקת מלאי השורש המקורי ולא directory שלם; תוכנו ו־output id צריכים להיקשר לסיכום. לאחר רישום שינוי מותר זה, כל שינוי אחר בקלטים או בתוצרי ההתאמה הוא כשל.

לפני כל batch יש לשמור את inputs, מזהה group, seed, scale, intent והגדלת calls_attempted. יש להבחין 2048 reserved, attempted, returned ו־persisted. כשל באמצע batch אינו מוכיח 0 או מספר תשובות מדויק. reference שחזר נשמר גולמית לפני prediction של אחד המודלים; לאחר מכן כל prediction נשמר מיד. כך כשל predictor אינו מעלים תוויות שכבר נצרכו. כתיבה, shape, dtype, finite או norm כושלים עוצרים בלי query חוזרת.

הפורמט צריך לקשור 512 שורות בכל scale 0.01/1/10/100, RNG יחיד 941033, אותה נקודה ואותה תשובה לשני המודלים. יש לדווח יחס Frobenius ו־maximum absolute לכל מודל, הבדל חתום וסימון regression בכל מדד, לא להסתפק ב־all-relative improved. קריטריון האיכות נשאר כל ארבעת יחסי המועמד <=0.001. נדרשים finite ו־nonzero reference norm ללא overflow/underflow שנעלם בנורמה; תגובות FP32 כפי שבוצעו יישמרו ללא עיגול נוסף שמסתיר מקור נתונים.

הבקר צריך להקפיא קוד, רישום, approved review, clean HEAD ולבדוק preflight process_and_children לפני קריאה או גיבוב של קלט מדעי. יש לאכוף deadline משותף של 180 שניות, worker 512MiB/BLAS1/2KiBפלט, ולדרוש קבלת tree גם לאחר העבודה. קבלת כשל אינה מתוקנת בהרצה חוזרת. before נשמר לפני parsing ו־after נבדק גם ב־handled failure; external kill עשוי למנוע finally, והיעדר after הוא unknown. terminal failed גובר על NPZ או metrics קיימים; אין quality success על סמך קובץ חלקי.

מינימום בקרות סינתטיות לפני הפעלת reference האמיתי:

- source/prereg/fit-audit/candidate/hash mismatch ו־bad preflight: factory לא נקרא.
- סמן קיים ב־fitroot עם output חדש: אין factory או query; כשל לאחר reservation משאיר סמן ואינו מאפשר retry.
- query זורק לפני החזרה או בקבוצה שנייה: attempted נכון, קבוצה קודמת שמורה, אין חזרה.
- predictor ראשון או שני זורק: reference שהוחזר נשמר; מספר query calls לא גדל מחדש.
- shape/nonfinite/zero-reference-norm/כתיבה כושלים: terminal failure ללא metric success.
- paired synthetic references עם יחסי שיפור והחמרות absolute שונות: כל ההבדלים וסימוניהם נשמרים; equality בקריטריון 0.001 לעומת nextafter מעליו.
- שינוי קלט בזמן ההערכה או worker scope לא מתאים: כשל, גם אם metrics/candidate artifacts קיימים.
- factory סינתטי מוודא reservation+intent לפני כל query, והבדיקות מזריקות RNG/inputs קטנים חדשים במקום להפיק את actual941033 inputs/reference. אין קריאה ל־public_weights או forward אמת בבדיקות.

קוד ההערכה צריך להישען רק על predictor ו־reference adapter, ללא import של optimizer/gradient או ניסיון התאמה חדש. גישה פנימית של adapter למשקלי fixture ציבורי לצורך ייצור תשובות אינה מותרת למבקר התוצאה, שישתמש רק במערכים שנשמרו וב־forward של המועמדים. seed זה הוא הערכת development ציבורית מוגדרת מראש, לא sealed hidden-teacher ולא הוכחת שקילות כללית. השער למודל הפרטי נשאר סגור.
