סקירת artifact transport ראשונית — לא פרוטוקול ניסוי

הליבה מייצרת root ו־NPZ basenames בלעדיים, מגבילה את הפלט הסדרתי לפני כתיבת shard, שומרת reservation הכולל manifest overhead ואינה משחררתו לאחר כשל כתיבה. shard/file/manifest partial נשאר כראיה; write exception הופך את store ל־terminal. verify_manifest מגביל manifest ל־1MiB, מאמת schema/גודל/שם/גיבוב streaming ומלאי מדויק ללא decompression. יצירת נתונים מדעיים, query accounting, runtime/controller ומגבלות RAM/זמן אמיתיות אינם בשכבה זו.

ממצאים לפני חתימה:

1. finite scan קדם ל־payload nbytes guard. ndarray broadcast/strided עשוי לייצג payload עצום בלי להקצותו, ואז np.isfinite מקצה מערך boolean עצום. פרוב sentinel בטוח הוכיח כניסה ל־finitecheck לפני reject של payload. יש לאכוף ndarray/dtype/nonempty/nbytes תחילה ואז finite scan ו־serialization.
2. json.loads ללא duplicate hook מאפשר keys סותרים בתוך אותה רשומת manifest. פרוב הוסיף name מוקדם שונה ו־verify קיבל את ה־name האחרון. יש לדחות duplicate keys כמבנה malformed; duplicate record names כבר נדחים בנפרד.

גבולות נכונים שיש לשמר: pre-serialization nbytes אינו גודל NPZ; overhead צריך reject נוסף לאחר serialization ולפני disk. failure reserve אינו actual bytes-on-disk מדוד. עד1GiB משותף כולל1MiB עתידי למסמכים חייב להיאכף גם ברץ העתידי; שכבה זו מגבילה רק store. Hashes אינם אימות מספרי. ההבטחה אינה כוללת concurrent adversarial writer, fsync, transactional durability או hard RSS.

הממצאים והפרובים משתמשים בערכים literal/arange/view בלבד. אין RNG, registered seeds, actual teachers/queries, consumed941xxx או private assets. בדיקת full-profile transport שכבר עברה אצל ההורה אינה חוזרת ללא צורך; פרוב עצמאי יכול לבדוק shards בגודל/צורה מייצגים יחד עם מסלולי הכשל החדשים.

חתימה סופית לאחר תיקון: שני הממצאים נסגרו בפרוב עצמאי. oversized broadcast view של מיליארד רכיבים נדחה לפי nbytes לפני isfinite-sentinel, ללא הקצאה או scan; duplicate JSON key נדחה. פרוב נוסף אימת NPZ header overhead המכשיל לפני disk למרות payload בגבול; basis shape528x528 עם arange נשמר ונטען זהה בביטים, serialized2,235,022bytes; verify_manifest הצליח כאשר read_bytes לכל NPZ נאסר, כלומר גיבוב streaming ללא read-all. short manifest append לאחר כתיבת shard שמר reservation, הפך store ל־terminal וחסם retry; verifier דחה את הקידומת. manifest symlink נדחה לפני shard והיעד החיצוני נשאר ללא שינוי.

MAX_RECORDS הוגדל ל־8192 ו־manifest cap1MiB נשאר עצמאי: אין הבטחה ש־8192רשומות בשמות כלשהם נכנסות בפועל, ואין אישור combined inventory של הניסוי. שלוש הצורות המקסימליות נבדקו אצל ההורה; הביקורת העצמאית בדקה צורת basis מייצגת ולא הריצה מחדש את כל preflight הקיים. אין כאן קבלת process-tree או עמידה נמדדת ב־180s/512MiB של ניסוי עתידי.

אושר component source בלבד לפי source-review.json. אין אישור data runner/controller/experiment, יצירת מורים רשומים או קריאות. broad protocol בתור ו־private gate סגור.
