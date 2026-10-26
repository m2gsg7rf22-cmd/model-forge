סקירת ליבה — השערות דרגה מתצפיות בלבד

המקור rank_selection.py מקבל שלוש מטריצות בלבד. אין בו paths, מודלים, teacher/width metadata, RNG, fitting או קריאת קלטים. הסקירה לא נגעה במטריצות הניסוי השמורות, ב־registered seeds או בנתוני941xxx. אישור core אינו אישור loader/pipeline/controller או actual analysis.

כללי החישוב נבחנו: SVD מוגן באמצעות globalmaxabs scale, fullp spectra עם structuralzero padding, rightvectors דקים; לצורכי eligibility וגאפים הסינגולרים מנורמלים לאחר מכן ל־leading1. floor=64epsmaxshape הוא מוסכמה מספרית יחסית ל־leading ולא noise bound. strict1e-10 נשאר baseline נפרד. ריכוז הגאפים הוא max Richardson או maxmin בשלושה מבטים תחת maximum pairwise prefix RMSsine<=0.01. בחירה ב־logscore עם exact computed ties בוחרת אינדקס קטן ביותר; אין טענה על שוויון ספקטרלי אידאלי.

נמצא boundary defect ותוקן: diag(10,1) נתן reportedratio10 אך log-difference מעט קטן מ־log10 ולכן לא נבחר. המקור המתוקן משווה boundedratio>=10 לפני log(ratio), ללא הרחבת tolerance. פרוב עצמאי אימת nextafter מתחת/בדיוק/מעל10 -> ranks[null,1,1]. mask לא־בוליאני/nan נדחה לאחר חיזוק guard; select מייצר בעצמו masks בוליאניים.

נוסחת prefix distance נכונה עבור בסיסי שורות אורתונורמליים: ||P_A−P_B||F/sqrt(2k)=sqrt(1−sum_{i,j<=k}|<v_i,w_j>|²/k). פרוב מפורש עם projectors הראה הפרש מרבי7.216449660063518e-16. סיבוב קטן בשלושה מבטים קיבל rank1, סיבוב גדול דחה stable; largest-gap עדיין הוחזר כפי שנרשם. eligible top בכל שלושת המבטים מונע שימוש ב־null-vector prefixes כ־stable candidates, אף שהדיאגנוסטיקה המלאה מחשבת גם אותם.

נבדקו scale1e±200 נפרד לכל view, row-orthogonal mixing control לאחר leading normalization, exact computed tie, zero/full/rowlimited/p1, invalid/nonfinite dtype/shape, ו־spiky fullsignal/commonlowrankbias. בכל הבחירות certificate=false. structuralzero-next בגבול rows<p עשוי ליצור candidate rowrank עם clipped-next; זו תקרת תצפיות, לא אומדן רוחב אמת. near-machine projector cancellation ו־floor clipping נשארים חישובים מקורבים, לא forwarderror/rank certificates.

הדוגמה observed=diag(1,1e-4,1e-4) יכולה להיות fullrank exactsignal או lowrank+commonbias; שלושת המבטים זהים וה־stable heuristic בוחר1 בעוד strictbaseline full3. זו המחשה למה stability לבדה אינה יכולה לזהות intrinsicfullspace או לאשר שאין signal ב־tail. אין הבטחה לגלות מצב כזה מתצפיות בלבד; לא נבחר כלל מנצח על בסיס נתונים ממשיים.

לפני actual saved-data analysis נדרשים source-reviewed whitelist loader, sealing כל16בחירות לפני context/reference/truerank/control reads, פלט מלא ומחויבויות, source/controller/prereg/cleanfreeze וניטור משאבים. known/half/double metadata controls צריכים לשמור requested/effective k ו־clipping, כי controls שגויים יכולים להתלכד אחרי cap. ההפרדה היא engineering view guard, לא OS isolation או blind public dataset. השער הפרטי סגור והמדע הרחב פתוח.
