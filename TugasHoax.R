library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('BJ Habibie Dikabarkan Meninggal.',
          'Dalam postingan tersebut disampaikan bahwa BJ Habibie sudah didampingi anaknya.',
          'Sebelumnya BJ Habibie tengah menjalani perawatan di Munich Jerman.',
          'Presiden ketiga Indonesia itu didiagnosis mengalami kebocoran pada klep jantungnya.',
          'Presiden Joko Widodo sempat menghubungi BJ Habibie secara langsung dan berbincang sejenak.',
          'Melalui pembicaraan tersebut, Presiden menyanggupi permintaan Habibie yang menginginkan adanya tim dokter kepresidenan dan Paspampres untuk hadir di Jerman saat dilakukan tindakan medis.',
          'Untuk mendampingi Habibie selama dilakukan tindakan medis.',
          'Presiden Joko Widodo sudah mengutus Prof. Dr. Lukman Hakim, SpPD-KKV, SpJP, Kger, seorang spesialis jantung dan pembuluh darah dari tim dokter kepresidenan, untuk berangkat ke Jerman, termasuk anggota Paspampres juga diberangkatkan.',
          'Penyakit jantung yang membuat presiden ketiga Indonesia yang membuat meninggal.',
          'sebelum pergi ke jerman pak habibi berpesan seolah mau meninggal.',
          
          
          'Kondisi kesehatan Presiden ketiga Republik Indonesia, BJ Habibie semakin membaik.',
          'Hal itu terjadi setelah mendapatkan perawatan di rumah sakit di Munchen, Jerman.',
          'Eyang Habibie sudah merasa lebih sehat tapi masih menjalankan pemeriksaan dan istirahat di RS di Muchen.',
          'Meskipun sudah merasa lebih sehat, The Habibie Center tetap meminta doa dari masyarakat Indonesia untuk kesehatan BJ Habibie.',
          'Presiden ketiga Indonesia tersebut sudah di kabarkan dokter kondisinya terus membaik.',
          'Melalui Menteri Luar Negeri, Presiden juga telah menginstruksikan kepada Duta Besar Republik Indonesia di Jerman untuk terus memantau kondisi terkini dari Habibie dan melaporkan langsung kepadanya.',
          'Selain itu, dirinya memerintahkan Menteri Sekretaris Negara untuk memastikan bahwa pemerintah mampu memberikan pelayanan terbaik dan menanggung seluruh biaya perawatan Presiden RI ke-3 itu sebagaimana diatur dalam Undang-Undang Nomor 7 Tahun 1978 tentang Hak Keuangan/Administratif Presiden dan Wakil Presiden serta Bekas Presiden dan Wakil Presiden Republik Indonesia.',
          'Presiden telah memerintahkan untuk memantau dan memberikan pelayanan terbaik kepada Habibie.',
          'Presiden sendiri berharap agar B.J. Habibie dapat kembali beraktivitas seperti sedia kala. Melalui sambungan telepon sore ini, ia bersama dengan seluruh rakyat Indonesia juga sekaligus mendoakan kesembuhan beliau.',
          'Kita semua di Indonesia, seluruh rakyat Indonesia, mendoakan Bapak. Semoga segera sehat kembali, bisa beraktivitas dan kembali ke Indonesia.')

corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)
train <- cbind(train, c(0, 1))
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)
data
train
# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

# Check accuracy on training.
predict(fit, newdata = train)

# Test data.
data2 <- c('Entah siapa yang memulai menyebarkan, namun isu tersebut berkembang dengan cepat.',
           'Sejumlah pengguna twitter pun seakan-akan berlomba menyampaikan ucapan belasungkawanya atas meninggalnya Presiden Habibie tersebut.',
           'Habibie dikabarkan meninggal setelah sebelumnya kritis di sebuah rumah sakit di Jerman.',
           'pusat penelitian yang dibangun oleh Habibie, yakni The Habibie Center, melalui akun twitter resminya, membantah kabar meninggalnya BJ Habibie tersebut.',
           'Senada dengan itu, artis Melanie Soebono yang merupakan cucu Presiden Habibie, juga membantah kabar tersebut.',
           
           'alam keterangan yang dituliskan The Habibie Center, disebutkan bahwa B.J. Habibie dalam kondisi sehat walafiat, dan sekarang sedang berada di Jerman.',
           'Alhamdulillah Bapak BJ Habibie dalam keadaan sehat walafiat. Beliau masih di Jerman sesudah merayakan Tahun Baru dengan cucu-cucu beliau.',
           'amun klarifikasi akun Facebook The Habibie Center sedikit membuat banyak orang terkejut.',
           'Pasalnya, saat kabar tersebut berhembus, Habibie malah dikatakan menghadiri sebuah acara penghargaan.',
           'Tadi malam beliau sangat senang ngobrol dan tertawa lepas dengan Reza dan Pandji LIVE dari Kediaman di Patra Kuningan di acara Indonesia Box Office Movie Awards di SCTV.')

corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)

