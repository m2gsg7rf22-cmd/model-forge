סקירת ליבה מתמטית עצמאית — scaling_domain_math

הסקירה מוגבלת ל־pure math המוזן במערכים מפורשים. אין אישור runner, פרוטוקול איסוף, יצירת מורים או ניסוי רשום. אין RNG, שאילתות, fitting או קריאת נתוני941xxx/684xxx בביקורת.

נבדקו נוסחת ההסיאן עם layout probe/output/input/input, חבילת svec איזומטרית ל־Frobenius, אטומים טבעיים בלתי מנורמלים בסדר interleaved, מרחב שורות דרך right singular vectors, סף strict1e-10, padding structuralzeros ל־p, ו־gap null עבור zero/full/zero-next. הסטנסיל משמר ארבע פינות וסדר חישוב עם effective diagonal2h; Richardson תואם h מול2h.

ממצא שהתברר ונסגר: direction_residuals החזיר false0 כאשר residual לא אפס בסדר1e-200 ונורמת ריבועים ירדה לאפס. המקור המתוקן מחשב נורמה בקנה מידה של כל שורה ומסרב ל־outer/projection arithmetic underflow המזוהה. פרוב עצמאי אימת residual1e-200 שמור ודחיית outer underflow של direction[1,1e-200].

פרוב חדש וקטן עם ערכים מילוליים בלבד אימת Hessian כנגד complex-step של Jacobian לוגיסטי עצמאי, max absolute error8.326672684688674e-17; אטומים ו־Frobenius inner product ידניים; strict nextafter על סף הדרגה; full/zero/deficient spans; נורמות זעירות; quartic stencil עם diagonalbias8h² וביטול Richardson; SVD global scaling1e-200/1e200. לא חזרו על בדיקות experiment או transport קיימות.

גבול חשוב: analytic_hessian הוא contracted FP64 reference מקורב. הוא רשאי להכיל computedzeros מהתכווצויות, saturation או underrepresentable products. למשל g=u=1e-100,d=1e-200,x=0 נותן אפס, בעוד הערך האידאלי1e-400 אינו ייצוגי. אין לראות ב־errstate הוכחה שכל contraction BLAS/einsum מזהה underflow, ואין להפוך computedzero ל־exact ideal rank certificate. תיחום חוזה זה הוסכם עם ההורה במקום הוספת חשבון מדויק בלתי נחוץ. גם projection/SVD הם נומריים בספים קבועים, ולא exact-membership או forward-error proof.

מגבלות נוספות: signature מצפה למערכי FP64 אך חלק מהעוזרים ממירים ערכים ל־FP64; רץ עתידי חייב לאכוף schema/shape/caps ומשמעות ערוצים. הליבה אינה מנהלת תקציבי query, reservation, דיסק, memory או deadlines. יש להפריד verifier privileged atoms/Hessian מ־learner observations. שייכות למרחב מלא אינה שחזור שערים, וחסם2m אינו שוויון או זיהוי. המשימה הרחבה נשארת פתוחה והשער הפרטי סגור.

החלטת מקור מדויקת וגיבובי תוכנית/בדיקות/מסמכים נמצאים ב־source-review.json לאחר הקפאת הבתים. תוצרי הפרוב ב־probe.py ו־probe-results.json חיצוניים למאגר.
