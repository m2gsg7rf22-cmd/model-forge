# ביקורת רכיב חשבונאות מלאי לרוחב משתנה

אושר רכיב fresh_rank_inventory בלבד על הבתים החתומים, לא פרוטוקול/מקור מלא או יצירת נתונים/teacher/RNG/query/fit/הרצה. אין NumPy, filesystem או model imports במקור; הוא מחשב shapes ומספרי integer ומשתמש במבניPython קטנים, ללא הקצאת מערכים מדעיים. פרופילי הרוחב הם metadata לתכנון מיוחס, וה־FDView אינו מייבא אותם.

## אימות בלתי תלוי

הספירה נגזרה מחדש מפרופיל: לכל מקרה truth=1context+32inputs+32responses+2reference=67shards, public=6, estimator=2, evaluator=1. ב־24 מקרים מתקבלים1608/144/48/24 וביחד1824. 24×16×2 נותן768 callbacks, ו־6×16×2×4×Σp נותן545280 vectors.

נגזרו בנפרד מספרי scalar בכל שמונת סוגיschema לכל12teachers/שני ערוצים, כולל4views שלH, ששת שדותbases, עשרתchoices ואחד־עשרevaluation. raw input/output כל אחד119,193,600bytes ו־Hviews115,015,680bytes. ב־n32/m96 בסיסthin הוא512×528, בעודfullspectrum/projector משתמשים528; atoms192×528 ו־gate_residuals6×96. אלה גדלים שונים ונשמרו נכון.

החסם לכלmanifestrecord הוא188bytes לפיbasename84תווים וסריאליזציה compact של ShardStore. מקסימוםmanifesttruth302304bytes ו־public27072bytes עומדים בתקרות1MiB/128KiB. לחסם ה־NPZ נוספים4096bytes לכלfield; עם שמותASCII≤64, עד4dimensions חיוביים, payload≤8MiB ו־FP64 פשוט, זהו מרווח שמרני מעלNPY header+ZIP overhead. נבדק חישוב ה־header/name overhead עבור כלfield בפרופילי הייצור. serializers עתידיים עדיין חייבים למדוד ולהגביל את הבתים בפועל; אין טענה כללית לכלפורמט/גרסתNumPy.

| store | shards | payload bytes | array upper bytes |
|---|---:|---:|---:|
| truth |1608|271190656|278661760|
| public |144|86261760|86851584|
| estimator |48|85336368|86909232|
| evaluator |24|143328|1224672|

maxarchivebound=6,785,056bytes. שבע משפחותpreflight+manifest=17,591,940bytes. סכום כלstores/manifest/preflight ורזרבתmetadata8MiB הוא479,970,708bytes, קטן מ־1GiB; minfree המתוכנן2GiB. זו תוכנית אחסון, לא הוכחת זמינותdisk/RSS/TREE או מספרקבצים שלrunner עתידי.

## ממצאים שטופלו

הפרוב הראשוני הראה ש־shape=list ניתן לשינוי לאחר אימות, ושמותfields ארוכים/לא־בטוחים אוfile/allow_pickle התקבלו. תוקן snapshotdict, regexASCII≤64 ושמותלא־שמורים, exacttuple עםintegerpositive/nonbool, ולאחר מכןMappingProxy. בפרוב נוסף shape של2000ones התקבל אף ש־NPYheader היה6051bytes וחרג מהנחתoverhead; תוקן1≤ndim≤4. כלגבולות הבנאי כעת עברו בפרוב עצמאי, לרבותimmutablemappingcopy. אין ממצא חוסם שנותר.

תוצאותRED נשמרו במפורש. ספירת הפרופילים העצמאית נעשתה לפני שינויי הבנאי בלבד; הנוסחאות/פרופילי הייצור לא השתנו. לאחרהתיקון נבדקו מחדש רק גבולותהבנאי; לא חזרנו עלroundtrip7maxfamilies שעבר אצל ההורה. בדיקותההורה18/mypy1/Ruff2 ו־serialization7families מדווחות בנפרד ולא מוצגות כבדיקות שהבודק הריץ.

ששת מקורותmath/store/rank_selection/IO/glu32/antipodal ו־FDView הקודם אומתו ללאשינוי; נבדקAST שה־FDView אינו מייבא inventory. תוצאותהפרובים, דוחזה והחתימות המדויקות מפורטים ב־source-review.json. לא נקראו actualownedobservations/private/941xxx ולא הופעלוRNG, teacher, forward, query אוfit.

## מחוץ לאישור

המקור אינו אוכף combinedcap, קבציmetadata/temporary/engine requests, chronology שלcommitments אוall24seal. מלאי זה חייב להיאכף ב־runner/controller/source עתידי, יחד עם actualdisk≥2GiB ו־TREEpreflight/180s/512MiB/BLAS1/2KiB. הוספת artifact אוformulaauxiliary שלא תוארו תחייב inventory/source review מחדש. אין הגדרתscientific energy formula מתוךשםarray בלבד. fullfreshprotocol/source/RNG/teacher/data/run נשאריםfalse; privategateclosed ואין טענת שחזור אוcertificate.

## תיקון integration של תקרותstore

נוסף לכלstore שדה shard_limit_bytes כמקסימוםbound שלschema באותוstore. זאת כדי ליצור כלShardStore עםcap קטן ומדויק יותר במקוםdefault8MiB, שאינו חוקי כשה־evaluator totalcap הוא1,229,184bytes.

בפרוב עצמאי נבנו ארבעהstores ריקים בלבד, ללאמערכים: truth/public מקבליםshardcap2,101,248bytes; estimator6,785,056; evaluator57,280. totalcaps הם278,964,064 /86,878,656 /86,918,256 /1,229,184 בהתאמה. כולם חוקיים מול הבנאי הקפוא. אומת שכלscientific count/payload/manifest/global/preflight total נשאר ללאשינוי; האישור COMPONENT ONLY והדגליםfalse של fullprotocol/data לא השתנו. החתימה הקודמת0047 והפרובים נשמרו, ולא בוצעquery/teacher/RNG/fit אוresourcepreflight. בדיקות ההורה כעת19/mypy1/Ruff2 מדווחות בנפרד.
