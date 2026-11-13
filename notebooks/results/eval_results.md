### Summary

| metric | value |
|---|---|
| Questions evaluated | 88 |
| Retrieval relevant (in-domain) | 91% (71/78) |
| Out-of-scope refused by the gate | 100% (10/10) |
| Retrieval relevant, field crops | 100% (23/23) |
| Retrieval relevant, vegetables | 100% (19/19) |
| Retrieval relevant, fruit trees | 100% (9/9) |
| Retrieval relevant, general | 81% (13/16) |
| Retrieval relevant, cross-language | 64% (7/11) |
| Retrieval | hybrid (bge-m3 + BM25), top_k=6 |
| LLM | not available |

### Per question

| id | lang | question | retrieved_source |
|---|---|---|---|
| q01 | ar | ما هو الميعاد المناسب لزراعة القطن في الأراضي القديمة؟ | نشرة-القطن2023_compressed-1.pdf, p. 7 |
| q02 | ar | ما أسباب ظاهرة الهياج الخضري في القطن؟ | نشرة-القطن2023_compressed-1.pdf, p. 13 |
| q03 | ar | ما هو أنسب ميعاد لزراعة القمح في الأراضي الجديدة؟ | NEW_wheat_new.pdf, pp. 6-7 |
| q04 | ar | كم كيلو تقاوي قمح يحتاج الفدان تحت نظام الري المحوري (البيفوت)؟ | NEW_wheat_new.pdf, p. 19 |
| q05 | ar | ما معدل تقاوي القمح عند الزراعة بالنقر على المصاطب؟ | NEW_wheat_new.pdf, pp. 10-11 |
| q06 | ar | لماذا ينصح بزراعة البرسيم الفحل قبل زراعة القمح؟ | القمح-قديم-2022_compressed.pdf, p. 15 |
| q07 | ar | ما هي أضرار دودة الحشد الخريفية على القمح وكيف تظهر الإصابة؟ | res4_-mohamed-abdel-hady-sayed-abdel-wahab.pdf, p. 31 |
| q08 | ar | متى يزرع بنجر السكر؟ | 1-نشرة-بنجر-السكر_compressed.pdf, p. 5 |
| q09 | ar | متى تجرى عملية الخف في بنجر السكر؟ | 1-نشرة-بنجر-السكر_compressed.pdf, pp. 7-8 |
| q10 | ar | ما هو ميعاد زراعة القصب الخريفي؟ | نشرة-االقصب-2023-copy_compressed.pdf, pp. 5-6 |
| q11 | ar | كم طنا من التقاوي يحتاج فدان قصب السكر؟ | نشرة-االقصب-2023-copy_compressed.pdf, p. 9 |
| q12 | ar | ما معدل تقاوي دوار الشمس للفدان؟ | نشرة-دوار-الشمس.pdf, p. 3 |
| q13 | ar | لماذا يجب توفير خلايا النحل بجوار حقول زهرة الشمس؟ | نشرة-دوار-الشمس.pdf, p. 9 |
| q14 | ar | ما هو أفضل ميعاد لزراعة السمسم؟ | نشرة-السمسم-copy_compressed.pdf, p. 3 |
| q15 | ar | متى وكيف يتم حصاد السمسم؟ | نشرة-السمسم-copy_compressed.pdf, p. 8 |
| q16 | ar | كم كيلو تقاوي يحتاج فدان الذرة الرفيعة؟ | الذرة-الرفيعة-نهائي_compressed.pdf, p. 3 |
| q17 | ar | متى تزرع الذرة الرفيعة في محافظات الصعيد؟ | الذرة-الرفيعة-نهائي_compressed.pdf, p. 3 |
| q18 | ar | ما أنسب ميعاد لزراعة الكتان؟ | الكتان2020_compressed.pdf, pp. 4-5 |
| q19 | ar | كم كيلو تقاوي كتان يحتاج الفدان في أراضي الوادي؟ | الكتان2020_compressed.pdf, p. 10 |
| q20 | ar | ما فائدة التلقيح البكتيري لتقاوي الفول البلدي؟ | الفول-البلدي-نهائي_compressed.pdf, p. 6 |
| q21 | ar | ما نسبة الزيت والبروتين في بذور فول الصويا؟ | نشرة-فول-الصويا-طبع_compressed.pdf, pp. 1-2 |
| q22 | ar | كيف أعالج مرض البكانا في الأرز؟ | NEW_rice_diseases.pdf, p. 2 |
| q23 | ar | ما علاج مرض اللفحة في الأرز؟ | NEW_rice_diseases.pdf, p. 1 |
| q24 | ar | كم جراما من بذور الطماطم الهجين يحتاج الفدان؟ | NEW_tomato.pdf, p. 11 |
| q25 | ar | ما سبب عفن الطرف الزهري في ثمار الطماطم؟ | NEW_tomato.pdf, p. 2 |
| q26 | ar | متى يعتبر محصول البصل ناضجا وجاهزا للتقليع؟ | NEW_onion.pdf, p. 8 |
| q27 | ar | كيف أجهز البصل بعد التقليع حتى يتحمل التخزين فترة طويلة؟ | التداول-الأمثل-لمحصول-البصل.pdf, p. 2 |
| q28 | ar | ما درجة الحرارة المثلى لإنبات بذور الفلفل؟ | الفلفل-16_compressed.pdf, p. 5 |
| q29 | ar | كيف تتم تقسية شتلات الباذنجان قبل نقلها للأرض المستديمة؟ | الباذنجان-17_compressed_compressed_compressed.pdf, p. 10 |
| q30 | ar | بعد كم يوم من الزراعة تقطف ثمار الخيار؟ | زراعة-وانتاج-الخيار.pdf, p. 9 |
| q31 | ar | كم كيلو بذرة كوسة يحتاج الفدان؟ | نشرة-الكوسة.pdf, p. 4 |
| q32 | ar | ما أسباب ظاهرة الساق الأجوف في الكرنب؟ | الكرنب_compressed.pdf, pp. 11-12 |
| q33 | ar | ما علامات نضج ثمرة البطيخ؟ | البطيخ.pdf, pp. 15-16 |
| q34 | ar | بعد كم يوم تنضج ثمار الكانتلوب المزروعة تحت الأنفاق؟ | الكانتلوب.pdf, p. 7 |
| q35 | ar | كم كيلو تقاوي بطاطس يحتاج الفدان في العروة الصيفية؟ | البطاطس_compressed.pdf, p. 12 |
| q36 | ar | ما هو التنبيت الأخضر لتقاوي البطاطس ومتى يجرى؟ | البطاطس_compressed.pdf, p. 10 |
| q37 | ar | ما درجة الحرارة والرطوبة المناسبة لتخزين درنات البطاطس؟ | ticket_file6214bd8f64479-Pomme-de-terre-Ardna (1).pdf, p. 9 |
| q38 | ar | متى تزرع شتلات الفراولة الطازجة لإنتاج ثمار التصدير؟ | نشرة-الفراوله-طبع_compressed.pdf, p. 12 |
| q39 | ar | متى تجرى عملية أقلمة شتلات الخضر قبل نقلها من المشتل؟ | NEW_vegetable_nurseries.pdf, p. 21 |
| q40 | ar | ما مواصفات شتلة الخضر الجيدة؟ | NEW_vegetable_nurseries.pdf, p. 22 |
| q41 | ar | كيف أكافح حشرة التوتا أبسلوتا في الطماطم؟ | NEW_tomato.pdf, p. 32 |
| q42 | ar | كم طنا من السماد العضوي يضاف لكل هكتار عند زراعة البطاطس؟ | ticket_file6214bd8f64479-Pomme-de-terre-Ardna (1).pdf, pp. 4-5 |
| q43 | ar | ما أعراض نقص الزنك في أشجار الموالح؟ | الموالح_compressed.pdf, pp. 9-10 |
| q44 | ar | لماذا تتشقق ثمار الموالح؟ | الموالح_compressed.pdf, p. 10 |
| q45 | ar | ما أبعاد الجورة عند زراعة شتلات الموالح؟ | الموالح_compressed.pdf, p. 3 |
| q46 | ar | متى يبدأ موسم جمع المانجو في أسوان؟ | نشرة-المانجو-طبع_compressed.pdf, p. 16 |
| q47 | ar | ما هي ظاهرة المعاومة في أشجار المانجو؟ | نشرة-المانجو-طبع_compressed.pdf, p. 20 |
| q48 | ar | ما معنى عدم التوافق الذاتي في أصناف الزيتون؟ | شجرة-الزيتون-نشرة-10.pdf, p. 12 |
| q49 | ar | كيف يتم تقليم رأس النخلة للحد من الإصابة بالآفات؟ | نخيل-البلح-نشر.pdf, p. 18 |
| q50 | ar | لماذا لا تستخدم البذور في إكثار نخيل البلح تجاريا؟ | نخيل-البلح-نشر.pdf, p. 4 |
| q51 | ar | كم يوما تحتاج ثمرة البلح من التلقيح حتى الحصاد؟ | نخيل-البلح-نشر.pdf, p. 3 |
| q52 | ar | ما الذي يقيسه التوصيل الكهربي EC في التربة؟ | Agricultural_RAG_200_Pages_Arabic.pdf, p. 7 |
| q53 | ar | كيف أمنع الآفات من اكتساب مقاومة للمبيدات؟ | AGRI_011_PDF.pdf, p. 17 |
| q54 | ar | ما القيم التي أطلبها من معمل تحليل التربة؟ | Agricultural_RAG_200_Pages_Arabic.pdf, p. 6 |
| q55 | ar | ما هي مبادئ الإدارة المتكاملة للآفات؟ | AGRI_013.pdf, p. 46 |
| q56 | en | What are the main principles of conservation agriculture? | AGRI_014_PDF.pdf, p. 2 |
| q57 | en | How much arable land has the world lost over the past 40 years? | AGRI_014_PDF.pdf, p. 1 |
| q58 | en | How many people live in areas where land degradation causes crop yield gaps? | cd7067en.pdf, p. 114 |
| q59 | en | What climate and rainfall does banana need? | farmerbook.pdf, p. 80 |
| q60 | en | What are some examples of organic manure? | farmerbook.pdf, p. 73 |
| q61 | en | How much milk does a Murrah buffalo give per day? | 577137840-agriculture.pdf, pp. 9-10 |
| q62 | en | What is agricultural biodiversity? | Unit 1.pdf, p. 5 |
| q63 | en | How much irrigation water is saved for Egyptian clover by irrigating only to 2.5 cm above the soil surface? | 1-2.pdf, p. 21 |
| q64 | en | Does magnetic irrigation water change maize plant height? | 2.pdf, p. 3 |
| q65 | en | What counts as personal protective equipment when handling pesticides? | AGRI_011_PDF.pdf, p. 12 |
| q66 | en | What is integrated pest management? | AGRI_011_PDF.pdf, pp. 10-11 |
| q67 | en | What is seed treatment and why is it used? | farmerbook.pdf, pp. 121-122 |
| q68 | en | When should sugar beet be planted in Egypt? | 1-نشرة-بنجر-السكر_compressed.pdf, p. 5 |
| q69 | en | What is the best time to plant sesame? | نشرة-السمسم-copy_compressed.pdf, p. 3 |
| q70 | en | What causes blossom end rot in tomatoes? | NEW_tomato.pdf, p. 21 |
| q71 | en | How do I treat bakanae disease in rice? | NEW_rice_diseases.pdf, p. 1 |
| q72 | en | When does the mango harvest start in Aswan? | نشرة-المانجو-طبع_compressed.pdf, pp. 46-47 |
| q73 | en | When should fresh strawberry transplants be planted for export production? | نشرة-الفراوله-طبع_compressed.pdf, p. 3 |
| q74 | en | How many kilograms of seed per feddan do I need for single-cross maize hybrids? | نشرة-الذرة-نهائى-copy_compressed.pdf, p. 5 |
| q75 | en | When should olives be harvested for oil extraction? | شجرة-الزيتون-نشرة-10.pdf, pp. 99-100 |
| q76 | en | What is the recommended planting date for flax? | الكتان2020_compressed.pdf, pp. 4-5 |
| q77 | ar | كم لترا من الحليب تعطي جاموسة المورا يوميا؟ | — (nothing relevant) |
| q78 | ar | ما هي أمثلة الأسمدة العضوية؟ | NEW_tomato.pdf, p. 28 |
| q79 | en | Who won the 2022 FIFA World Cup? | — (nothing relevant) |
| q80 | ar | ما هي عاصمة فرنسا؟ | — (nothing relevant) |
| q81 | ar | كيف أبدأ مشروع تربية النحل لإنتاج العسل؟ | — (nothing relevant) |
| q82 | en | How do I start a tilapia fish farm? | — (nothing relevant) |
| q83 | ar | كيف أحصن الدواجن ضد مرض النيوكاسل؟ | — (nothing relevant) |
| q84 | en | How do I change a flat tyre on my car? | — (nothing relevant) |
| q85 | ar | ما هو أفضل هاتف ذكي يمكن شراؤه هذا العام؟ | — (nothing relevant) |
| q86 | en | Can you recommend a good recipe for chocolate cake? | — (nothing relevant) |
| q87 | ar | كيف أتعلم البرمجة بلغة بايثون؟ | — (nothing relevant) |
| q88 | en | What is the current price of Bitcoin? | — (nothing relevant) |
