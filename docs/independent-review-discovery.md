# ביקורת עצמאית של Discovery Lab וזיכרון המחקר

תאריך: 2026-10-21. הבדיקה נעשתה בידי סוכן אימות נפרד מסוכני המימוש, באותו פרויקט ובאותו משתמש מערכת הפעלה. זו ביקורת תוכנה ותהליך מחקר עצמאית בתוך הצוות; זו אינה ביקורת של מדען חיצוני או בידוד אבטחה בפני תהליך זדוני מאותו משתמש. לא נוצר, אומן או שולב המודל הפרטי; תנאי הקדם שלו לא הושלמו והשער נשאר סגור.

## פגמים ששוחזרו ותוקנו

1. **דליפה בין תנאי ניסוי.** בגרסה הראשונית כל תנאי התצפית התייחסו לאותו teacher ופורסמו יחד. התאמת polynomial לנתוני real_noiseless הציבוריים ושימוש בו בתנאי sign_only השיגו MSE ‏9.87048e-29 ושגיאת coefficient מרבית 6.35603e-15. לכן תצורת הפרוטוקול אפשרה שחזור משרעת שלא הגיע מתצפיות סימן. התיקון מגריל teacher פרטי נפרד לכל תנאי. אותו ניסיון דליפה בהרצת הביקורת המתוקנת נתן MSE ‏4.40813. זהו probe המראה שהדליפה הספציפית הוסרה; עיקר ההצדקה להפרדת המידע הוא בדיקת קוד היצירה והקלטים, לא הסתמכות על מספר MSE אחד.
2. **חתימת דוח שהתיישנה.** בתחילה orchestrator שינה report.json לאחר שה-verifier חתם את ה-hash שלו. הביקורת שחזרה אי-התאמה. כעת verification-core.json נשאר בלתי משתנה, marker הפרטי חותם אותו והדוח הסופי מפנה לאותו hash ומוסיף מעטפת מדידת משאבים. שלושת הערכים התאימו בבדיקה עצמאית.
3. **מחיקת manifest ביטלה אימות משימה.** verify_manifest החזיר מילון ריק כשהקובץ חסר. כעת הקובץ חובה והבדיקה נכשלת לפני שינוי checkpoint.
4. **משימה פרטית הוצגה כזמינה למרות שער סגור.** השלמת תלות המשימות הספיקה להצגתה ב-eligible_tasks. כעת private_luna_model ומשימות המסומנות requires_private_model_gate אינן מוצגות כזמינות; מצבי running/completed שלהן נדחים כל עוד השער סגור.

## ריצה עצמאית לאחר התיקונים

ה-probe השתמש בממשק run_discovery, ב-teachers שהוגרלו מחדש וב-seed ציבורי 17. שלבי prepare/solve/verify היו תהליכים נפרדים. בדיקת הביקורת השוותה בנוסף את הפולינום של truth למפת המונומים של learner לכל תנאי; מעריך ה-oracle משתמש בלולאת מונומים נפרדת.

| תצפית | MSE במבחן החתום | שגיאת coefficient מרבית | עמידה בסף exact המספרי שנרשם מראש |
|---|---:|---:|---|
| real_noiseless | 5.53632e-29 | 6.38378e-15 | כן |
| real_noise_0.0001 | 3.40261e-9 | 7.12845e-5 | לא |
| real_noise_0.1 | 0.00494621 | 0.112139 | לא |
| sign_only | 0.289362 | 1.20786 | לא |

משך: 0.215372 שניות. שיא RSS שנדגם לעובד: 38,813,696 בתים, תחת תקציב 512MiB/60 שניות. זו דגימה כל 10ms של עובד בודד, לא הבטחה לשיא רציף ולא צריכת מערכת מלאה. לא נעשו קריאות רשת, GPU או API בתשלום. Teachers שונים בין תנאים מונעים השוואה סיבתית מבוקרת של גודל הרעש לבדו; הטבלה מתארת תוצאות נפרדות ולא עקומת noise-scaling מוכחת.

ראיה מקומית: `/private/tmp/mf-independent-final-5ei9k0of/review-results.json`. אין בה teacher weights או labels של המבחן החתום. נתוני truth פרטיים נשמרים בנפרד ב-temporary directory ולא נועדו לפרסום.

SHA256 של המקור בעת ריצה זו:

- discovery.py: `7dd1d5ec86cde0206bd6dafe6a647a75bf4ed20a8c098b569c14e3b2583f6d41`
- discovery_oracle.py: `e4f6255050504d52ef966af8bced79e61e69d322a2d64db237dd204c37103873`

ניסיון evaluation שני נדחה ב-FileExistsError. השוואת hash של core, marker ו-report עברה. בדיקה בסביבת Git זמנית הראתה ש-manifest חסר נדחה; משימה פרטית עם כל תלויותיה completed עדיין אינה זמינה; ניסיון לסמנה running או completed נדחה. בנפרד, 21 בדיקות test_research_memory עברו, כולל crash לאחר append לפני checkpoint, ledger קרוע/משונה, בעלות lock חיה ו-PID ממוחזר. אין מכאן הבטחה לעמידות בכל קריסת מערכת או למניעת החלפת כל ההיסטוריה בידי בעל הרשאות כתיבה; hash chain אינו חתימה של גורם חיצוני.

## ההסקה המדעית המותרת

למשפחה הידועה של פולינום מדרגה לכל היותר 2 בארבעה קלטים ושני פלטים, מטריצת מונומים בעלת דרגת עמודות מלאה מזהה את המקדמים הקנוניים באריתמטיקה מדויקת. הניסוי תומך בשחזור float64 לפי ספים מוגדרים. הוא אינו מוכיח byte-identical recovery, אינו מזהה ארכיטקטורה לא ידועה ואינו משחזר Transformer או מודל שפה סגור.

אי-זיהוי משרעת מ-sign-only נובע מהעדות האלגברית sign(2f(x))=sign(f(x)) לכל x; לכן קיים לפחות חופש קנה מידה חיובי. אין להסיק ממנה שאין דרך ללמוד את גבול ההחלטה או התנהגות מועילה. אי-עמידה בסף exact תחת רעש בריצה מסוימת אינה משפט אי-אפשרות אוניברסלי.

Train ו-validation ציבוריים; test inputs/labels ו-teacher entropy אינם נגזרים מה-seed הציבורי ואינם מועברים ל-solver. כשל evaluation צורך את הסימון לפני קריאת truth, והניסיון אינו ניתן לחזרה דרך אותו ממשק. תהליך עם גישת כתיבה מלאה לאותם קבצים יכול להסיר סימונים, ולכן זו הגנת פרוטוקול ויושרה בתהליך העבודה, לא security sandbox.

## נושא שנשלח להקשחה נוספת

בזמן כתיבת הדוח verifier בדק התחייבויות, source hashes, צורת coefficients ואת שגיאת ה-test, אבל העתיק candidate validation scores ו-validation-retained מתוך solution. נדרשה בדיקה מחדש של validation והחלטת הבחירה לפי preregistration, או סימון מפורש של שדות אלה כ-self-reported. ביקורת תיקון תצורף להלן אם יושלם.

### אימות ההקשחה — הושלם

לאחר תיקון verifier נבנות ומותאמות מחדש שלוש החלופות מהתצפיות הציבוריות באמצעות מעריך המונומים הנפרד; scores, rank, condition number, בחירה, retention ו-coefficients נבדקים מחדש. הביקורת יצרה ארבעה פתרונות פגומים **לפני חתימת seal**, כך שבדיקת hash לבדה לא הייתה יכולה לזהות את השיבוש. כל ארבעת המקרים נדחו: score מומצא, בחירת שיטה שגויה, retention שגוי ושינוי coefficient. הסימון החד-פעמי נצרך גם בכשלים אלה.

לאחריהם נערכה ריצה חדשה של הפרוטוקול התקין במקור המעודכן: noiseless MSE ‏1.39089e-28 ושגיאת coefficient ‏7.10543e-15; ‏noise 1e-4 ‏MSE ‏2.05911e-9; ‏noise 0.1 ‏MSE ‏0.00318265; ‏sign-only ‏MSE ‏0.859015. רק התנאי ללא רעש עמד בסף exact המספרי. זמן 0.238158 שניות ושיא RSS שנדגם 39,190,528 בתים. אין להשתמש בהבדלים מריצת הביקורת הקודמת כמדד לשיפור: ה-teachers הוגרלו מחדש לצורך אימות התיקון, ולא נעשה tuning לפי תוצאות sealed tests.

ראיה מקומית נוספת: `/private/tmp/mf-independent-selection-e2vl9x5_/review-results.json`.

המקור המעודכן שנבדק: discovery_oracle.py ‏SHA256 ‏`57c9ac4fb01206ad160d56b04d87381973fb520ad3fc76e263027ee8da74581e`; ‏discovery.py נשאר `7dd1d5ec86cde0206bd6dafe6a647a75bf4ed20a8c098b569c14e3b2583f6d41`.

אין פגם ידוע פתוח בחמשת הנושאים הממוקדים שתוארו כאן לאחר בדיקה זו. אין בכך אישור לכלל המערכת, להשלמת מחקר שחזור אוניברסלי, לממשקי CLI/MCP/GUI שטרם נבדקו בסבב זה או לפתיחת משימת המודל הפרטי.

## ביקורת אינטגרציה ו-KV cache — סבב נוסף

נבדק מימוש ה-cache מול MLX-LM המותקן: prefill נעשה פעם אחת; בכל צעד נוסף מוזן token יחיד; לכל generation נוצר cache חדש. לא זוהתה בבדיקת הקוד שגיאת causal order או offset. cache.state מחזיר את האורך הלוגי שנצרך, ולכן cache_logical_bytes אינו טענה לסך ההקצאה הפיזית. נדרשות עדיין הבדיקות הנומריות המתוארות להלן לפני קביעה על מקור ההבדל בפלט.

ראיית root על SmolLM2-135M ב-BF16 מציגה פלט שונה בין cached/full-prefix החל מ-token index 15. זמני cached של כ-0.156 שניות לעומת full-prefix של כ-0.45–0.57 שניות הם תוצאות מקומיות על פלטים שאינם זהים, ואינם הוכחת optimization שומרת פלט או איכות. ברירת המחדל נשארת full-prefix. נדרש teacher-forced comparison על אותם prefixes וכן float32 control כדי להבחין בין numerical sensitivity לבין שגיאת מימוש. לא בוצעה ריצת GPU נוספת בידי סוכן הביקורת.

בדיקת transport אמיתית של MCP באמצעות subprocess ו-SDK עברה: זוהו 32 כלים; run_discovery/resume_research/benchmark_mlx_cache מספקים schemas מפורשים; קריאת resume_research אמיתית החזירה execution=report_only ו-private_model_gate=closed.

שני פגמי אינטגרציה נמסרו ל-root להשלמה ולא אושרו עדיין בסעיף זה: callback ה-GUI לבדיקת cache התעלם מבחירות prompt/token/memory/bits במסך; ו-Service.run_discovery שמר finding רק בהצלחה, כך שכשל worker לא יצר רשומת כשל מדעי בהיסטוריה. אחסון truth פרטי durable במקום temporary directory נמצא גם הוא בשינוי; אישורו מצריך בדיקה לאחר השלמה.

### אימות אינטגרציה ואחסון מתמשך — הושלם

בדיקת Service אמיתית הפעילה worker של Discovery. ה-truth נשמר כעת ב-private-research מחוץ לפרויקט הציבורי, בתיקייה בעלת mode ‏0700. התיקון מחליף את אחסון ה-tmp שתואר בסבב הקודם; הוא אינו מספק encryption או בידוד ממשתמש המערכת עצמו.

ייצוא ZIP בפועל כלל observations.json, solution.json ו-verification-core.json; לא כלל truth.json, private_teacher_seed או evaluation-consumed marker. הזרקת TimeoutError מכוונת ל-engine נשמרה כ-discovery-failed בהיסטוריה ונזרקה מחדש, כדי שממשק המשתמש יקבל שגיאה ולא הצלחה. כל זאת נבדק ללא GPU.

callback אמיתי ב-Qt offscreen נבדק מול שירות לוכד: prompt הכולל עברית, תקציב 7 tokens, זיכרון 1024MiB ו-quantization של 4bit הועברו במדויק מן המסך. בדיקה זו מאמתת UI dispatch; היא אינה מאמתת חלון Mac גלוי או הרצת GPU דרך כפתור.

ראיה: `/private/tmp/mf-review-integration-ufq63wmr/review-results.json`.

קוד teacher-forced cache comparison נבדק: שני המסלולים מקבלים אותו prefix שנבחר מ-full-prefix reference, ולכן ההשוואה מבודדת הבדלי logits גם לאחר divergence. float32 control ממיר weights ומעריך אותם לפני מדידה ודוחה שילוב עם quantized weights. לא בוצע GPU rerun עצמאי, ואין כאן מסקנה סופית על סיבת ההבדל ב-BF16; תוצאות ריצות root צריכות להישמר בנפרד.

## ביקורת מחקר attention

המשפחה הנבדקת היא softmax attention חד-ראשי קטן, ללא mask וללא bias. קלט X, מטריצת הסתברויות attention מלאה P וממד head ידוע הם תצפיות מותרות. משוואת log-ratio מחזירה את A=Wq Wk^T בתנאי full-column-rank design. אינה מחזירה את Wq/Wk המקוריים: gauge הפיך Wq'=WqR ו-Wk'=WkR^{-T} משאיר את A ללא שינוי.

נמצאו ותוקנו שני פגמים לפני אישור:

1. private_storage שהיה תת-תיקייה של destination ציבורי התקבל ויכול היה לחשוף truth בתוך artifact. כעת שני כיווני קינון נדחים לפני יצירת התיקיות; הביקורת אימתה זאת בשני המקרים.
2. הדוח הציבורי הכיל תחילה hashes ו-summary בלבד, ללא X/P או student matrix, ולכן לא ניתן היה לשחזר את ההתאמה הציבורית ללא גישה ל-truth פרטי. כעת authorized_observations והפתרונות clean/noisy כלולים בדוח הציבורי, עם commitments. לא נחשפים teacher weights או secret seed.

ביקורת נומרית עצמאית חדשה חישבה את A רק מן X/P/h הציבוריים, באמצעות log-ratio ביחס לעמודת ההסתברות **האחרונה** במקום הראשונה ובניית Kronecker design. בשלושת seeds ‏11/23/47, ההבדל המרבי מן student solution השמור היה ‏6.24500e-16, ‏3.46945e-16 ו-4.44089e-16. שגיאת A שהוערכה מול truth בפרוטוקול הייתה ‏6.10623e-16, ‏5.82867e-16 ו-2.32453e-16. שלושת התנאים ללא רעש עמדו בספים; noisy-probability controls לא עמדו בסף exact. gauge, row-shift ו-subspace controls תקינים בהיקף המוצהר.

משך ריצה: ‏0.397703 שניות. שיא RSS learner שנדגם: ‏33,865,728 בתים. SHA256 של attention_research.py שנבדק: `fe46e8ca9634f8fbf62353751d7d49d748ea1dd4cf05565b265df35715996595`.

ראיה: `/private/tmp/mf-review-attention-r58f26th/review-results.json`.

ה-output-only counterexample משתמש ב-value projection מדרגה אפס. זו עדות קיום לאי-זיהוי במשפחה הכוללת מקרה מנוון זה, ולא משפט אי-אפשרות עבור V הפיך ידוע או כל attention layer. שירות טקסט רגיל אינו מעניק גישה ל-P. לכן הממצא הוא שחזור מטריצה מורכבת במשפחה ידועה תחת גישה מפורשת להסתברויות; אינו שחזור attention מממשק טקסט, Transformer שלם או משקלים סגורים. תנאי הקדם למודל הפרטי עדיין לא הושלם.

### סקירת ראיית KV-cache controls של root

נבדק גם `/private/tmp/modelforge-mlx-kv-controls.json` שנוצר בריצת root בפועל, על SmolLM2-135M-Instruct revision ‏12fd25f77366fa6b3b4b768ec3050bf629380bac. source digest המדווח הוא `1f9d8effae7ad6b9e17ecbce076c831d92b175617ece9441e07ae4b8607aeec5`.

ב-native BF16, השוואת אותם prefixes מצאה max absolute logit error ‏0.5 ו-argmax disagreement בצעד 15; token parity בריצה החופשית נכשל. ב-float32, max error היה ‏5.34058e-5, לא נמצאו argmax disagreements, ושתי החזרות החופשיות שמרו על אותם 32 tokens. התוצאות תומכות ברגישות נומרית התלויה בדיוק החישוב; הן אינן מוכיחות שקילות אוניברסלית או שוללות כל פגם אפשרי על כל ארכיטקטורה וקלט.

בקרת float32 מדדה cached ‏0.249–0.254 שניות לעומת full-prefix ‏0.561–0.638 שניות על הפלט הזה, עם warmup וסדר מתחלף. אלה מדידות מקומיות באותו דיוק על אותו רצף, לא benchmark של איכות שפה או הבטחת speedup לכל context. המרת BF16 ל-float32 משנה גם את מסלול החישוב והפלט לעומת native; אין להשוות איכויות בין שני הדיוקים מן prompt יחיד. ברירת המחדל נשארה uncached. הביקורת קראה את הקוד ואת ראיית root; היא לא שחזרה את ריצת ה-GPU בנפרד.
