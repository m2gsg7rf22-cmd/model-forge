# ביקורת תכנון בלבד: אימות rank עם רוחבים משתנים

הטיוטה נבדקה ללא RNG, יצירת teacher, תצפיות, forward, fitting, גישה לפרטי או נתוני941xxx. התכנון המתוקן אושר כתכנון בלבד: הספירה והמסגרת המתמטית עקביות, והתיקונים הקריטיים להלן עוגנו בטיוטה העדכנית. אין כאן אישור מקור, ריצה או יצירת נתונים. חתימת טיוטת הביקורת מופיעה ב־design-review.json.

## ספירות והמרחב הנמדד

12 teachers, שני ערוצי compute ו־16 probes נותנים24 מקרים. בכל מקרה שני צעדי FD, וכל callback מכיל4p שורות בסדר sign-major ללא dedup. לכן batches=24×16×2=768; vectors=6×16×2×4×(10+36+136+528)=545,280. Richardson נגזר מאותן שתי רמות ואינו callback נוסף. reference אנליטי ובקרות כיוון אינם שאילתות forward נוספות, אך חייבים להישאר במסלול evaluator בלבד. יש להפריד reserved/attempted/returned/persisted ואת מניין callback ממניין vector.

גבול span האנליטי הוא לכל היותר min(2m,p), לא שוויון מובטח. עבור רוחב2/4/8 ב־n4 מתקבלים גבולות4/8/10; ב־n8 רוחב4/12/20 נותן8/24/36; ב־n16 רוחב8/24/48 נותן16/48/96 לפני degeneracy; ב־n32 רוחב16/48/96 נותן32/96/192. אין להשתמש בגבולות האלה כ־rank labels. העתקה מדויקת של g וגםu יוצרת יחידות mergeable באמצעות חיבור עמודות D ומורידה את גבול מספר היחידות האפקטיבי; אין להסיק original width. near-parallel עלול ליצור רכיבים חלשים בסדרים שונים ואינו קובע rank נומרי מראש.

16probes מצמצמים את מספר שורות התצפית ל־16n. ב־n32 אלו512 שורות לעומת528 עמודות; structural zeros ופערי next-sigma המגיעים מתקרת שורות חייבים להישאר מסומנים. זה אינו מדד חדש ל־full analytic span. בפרופילים המוצעים min(2m,p) שלn32 עד192 בלבד, כך שלא מתוכנן בו מקרה truth-full528. מקרים truth-full פוטנציאליים הםn4/m8 ו־n8/m20, אך התווית נקבעת רק מהייחוס הנומרי.

הרוחבים המשתנים אכן שוברים את יחסm=2n הקודם, ובכל זאת זו משפחה סופית וידועה עם שלושה רוחבים לכלn. אין להסיק בדיקה של רוחב כללי לא ידוע, מימד סיבתי או יכולת על משפחות שונות.

## חוזים שנדרשו בביקורת והשלכות למימוש

1. **הגדרה חד־משמעית ל־duplicate/near-parallel.** קבעו indices, סדר draw/overwrite, אילו שורותg/u מועתקות/מופרעות, האםD נשאר בלתי תלוי ובאיזה שלב מעגלים לערוץFP32. הביטוי gate-pair אינו מספיק. להעתקה mergeable יש להעתיק בדיוק את שתי השורותg ו־u. ל־near-parallel יש לציין בנפרד delta_g/delta_u, מקורן והאם1e-6 הוא שינוי מוחלט, יחסי לנורמה או ביחידותstd. perturbation שלstandard-normal×1e-6 מולweightstd.2 אינו שינוי יחסי1e-6. שמרו factors/perturbations ו־angle/norm diagnostics ב־truth בלבד; אין לנרמל בדיעבד לפי תוצאה.

2. **הגדרת precision/reference.** לכל ערוץ יש לקבע dtype של משקלים, inputs, activations ומכפלות/סכימות, לרבות batch partition. לערוץFP32 הגדירו במפורש smooth analytic reference ב־FP64 של המשקלים המעוגליםFP32, אם זו הכוונה כמו בניסוי הקודם; אין לכנותו derivative של הפונקציה הדיסקרטית המחושבתFP32. לערוץFP64 קבעו שימוש במשקלים המקוריים. הגדירו שיתוף probes/random directions בין שני הערוצים; אין draw סמוי או callback אנליטי המוסיף forward calls.

3. **גבול observation/truth נפרד.** אין להשתמש ב־legacy Inputs שמפרסר manifest מלא: גודל context NPZ ושמות/headers יכולים לחשוףm עוד לפני פתיחת context. נדרש root/manifest נפרד לתצפיות FD בלבד, עם IDs ניטרליים, shapes שתלויות רקn/output/probes ושלוש view matrices. truth manifest, context sizes/headers, generation seeds, perturbations, width/type assignment ודוחות teacher אינם parsed לפני אטימת כל24 הבחירות. hash-only בידי controller אינו קלט estimator. גם קוד loader חדש חייב למנוע leakage דרךסדר case IDs, filenames, source loop order ושדות summary.

4. **התחייבות assignment שאינה enumerable.** קבוצת permutations לשלושה רוחבים היא רק3!=6 לכלn, ולכן hash של mapping בלבד ניתן לניחוש. יש לפרסם התחייבות canonical mapping/config עם nonce סודי לפחות128bits או seed התחייבות בעל entropy מתאים, ולחשוף אחר אטימת24 הבחירות, יחד עם hashes. המקור הציבורי אינו יכול להכיל seeds/assignment קריאים ל־estimator ולהיקרא withheld. permutation נחוצה אם השמה אחרת נחשפת מסדר הטבלה; היא אינה OS security או statistical blinding. יש להבחין בין teacher-generator נפרד לבין מסלול estimator שבקוד הנבדק אינו פותח truth.

5. **כלל אי־ודאות קבוע לייחוס.** שמרו rank convention1e-10 אך הגדירו מראש מהו weak-spectrum uncertainty, למשל band קבוע סביב cutoff, אילו rescaling/representability guards נדרשים ואיך מסמנים את המקרה. אין לשנות cutoffs או להכריז בדיעבד 'uncertain' כדי להסיר כשל. leading singular value חיובי ובר־ייצוג נדרש ל־0<r; fullp משמעו sigma_p>1e-10 sigma_1 לפי אותה אריתמטיקה. ספקטרום0/nonfinite/underflow או uncertainty מפורש לא יהיו pass ולא יוסרו מהמכנה.

6. **כללי endpoint מלאים.** הקפיאו שני endpoints נפרדים ל־gap ול־stable על כל24 המקרים, ללא best-rule promotion. ב־0<r<p יש לדרוש rank exact וכלloss/extra/gate/random conditions יחד; ב־r=p יש לדרוש null. קבעו במפורש השוואת floating של minrandom>maxgate ללא שינוי margin בדיעבד. resource failure גובר על כל המדדים. ניתן לדווח pass-rule נפרד ו־overall-all-rules, אך כשל באחת אינו נעלם אם השנייה עברה. שמרו גם strict וכל בקרות הרוחב כמדווחים שאינם מחליפים את endpoint.

7. **מדדי אנרגיה שאינם מחליפים גאומטריה.** הגדירו במדויק per-component diagnostics, סדר singular vectors וטיפול ב־degenerate singular values. H-reference relative reconstruction residual ||H-H P||F/||H||F או retained weighted spectrum שונים מ־lost-subspace RMS. הם auxiliary בלבד ולא מצילים endpoint כושל. שמרו full spectra, threshold margins ו־requested/effective/clipped/collision של כל known/half/double width controls.

## משאבים ותאימות

נדרש loader/runner חדש: old context validation המגבילm≤64 לא מתאים ל־m96; אין לשנות את המקורות החתומים הישנים. הגדירו width bound96, n/output≤32, 16probes, source schemas לכלraw/derived array, dtype, NPZ field counts, safe names, failure/partial artifacts ו־global disk/report caps.

כחסם תכנון בלבד, אם כל signed input וכל output נשמריםFP64 פעם אחת, בכל אחד מהם119,193,600bytes של נתוני scalar גולמיים: Σ_n 6×16×2×4p×n×8. יחד238,387,200bytes לפני headers/metadata. ארבעה Hessian views מלאים fine/coarse/Richardson/reference מוסיפים115,015,680bytes (=24 במבנה6pern×16×n³×8×4), לפני spectra/bases/projectors/context/traces. אלו אינם inventory כולל או הוכחת1GiB. הפרדה לשני roots אינה פוטרת global count/cap.

ב־n32 batch יחיד4p×n ב־FP64 הוא540,672bytes לכלinput אוoutput. שלושת בסיסיSVD בגודל512×528 דורשים6,488,064bytes לפני spectra/headers; שלושה projectors דורשים6,690,816bytes לפני diagnostics. אלה מתאימים בקירוב לתקרות הישנות, אך יש להוכיח מספרfields, serialization cap ומספרshards/manifest עבור source החדש. אין להריץ preflight היוצר registered teachers; בדיקתworst-shape literal/zero/arange תספיק לתובלה ולמשאבים לפני saved-data/generation.

המקור/פרוטוקול העתידי חייבים לכלול source/dependency/assignment commitments, clean commit, exclusive global marker לפני RNG/teacher/query, pre-data TREE+disk+transport gates,180s/512MiBpolledTREE/BLAS1/2KiBstdout, כלinput לפני callback וכלreturn לפני נגזרות, counters חלקיים, all24seal-beforetruth, terminal checkpoint ואז audit עצמאי בלי requery. אין אישור לבצע אותם בטיוטה זו.

## מסקנת הביקורת

הספירה נכונה וההרחבה לרוחב משתנה היא challenge מוגדר טוב יותר מהישענות עלm=2n, בכפוף לסגירת הסעיפים לעיל. גם endpoint מוצלח יהיה רק subspace evidence סופי עלowned isolated bias-free SwiGLU. הוא לא מראה full-function improvement שלcandidate7f7d, אינו משנה כשל941033, אינו מוכיח חילוץ גורמים, rank theorem מותנה, blind reconstruction או יכולת universal. שער המודל הפרטי נשאר סגור.

## אימות התיקון וחתימת התכנון העדכנית

הטיוטה העדכנית סוגרת את ממצאי התכנון הקריטיים: duplicate מעתיקg/u האחרונים מהראשונים ומשאירD; near-parallel מוסיף1e-6×.2×noise נפרד לכל אחת משתי השורות ללא normalization, לפני rounding לערוצים. reference שלFP32 הוא smoothFP64 על factors מעוגלים, ולא נגזרת ערוץFP32.

נדרשobs-onlyroot/manifest בלי context headers/bytes, IDs ניטרליים והשמה מוסתרת מפורשות; seed+mapping מקובעים בהתחייבותnonce≥128bits לפני generation, truthbytes מקובעים אחרי persistence ולפני estimator, ו־reveal/open רק אחרי all24seal. ידיעת אופציות הפרופיל נותרת ציבורית ואין טענת blinding. ה־validator הישןm≤64 אינו מתאים ל־m96.

כלל sensitivity קבוע ל־1e-11/1e-9 חל כעת על כלreference לא־אפס, גם primaryfullp; שינויrank ייחוס נכשל גם אםnull חזר. בפרטσp/σ1=5e-10 לא יכול לעבור כ־full כשה־side rank1e-9 חסר. אפס ואי־ודאות אינם הוצאה מהמכנה. כל מקרה בכל אחת משתי השיטות נשאר ב־endpoint הכולל. אין ממצא תכנון קריטי פתוח בסנפשוט החתום.

ההמלצות לעיל שנותרו לפרטיsource—fullinventory, exact per-component energy formula, precision/batch source pins, strictschemas/counters/partialIO ו־transport/resource controls—הן דרישות למימוש ולביקורת הבאה, ולא דברים שנבדקו או אושרו כאן. אין fullsource/teacher/RNG/data/actualanalysis approval. הטיוטה עודנהDRAFT ולא executable.
