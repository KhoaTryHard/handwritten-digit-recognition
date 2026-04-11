# 2.4 Phuong Phap Nghien Cuu

De tai duoc trien khai theo huong nghien cuu thuc nghiem tren bai toan nhan dang chu so viet tay. Quy trinh nghien cuu bat dau tu viec lua chon bo du lieu chuan, xay dung mo hinh nen, tinh chinh mo hinh theo tung mien du lieu va danh gia ket qua tren du lieu muc tieu. Trong de tai nay, nhom su dung ba nguon du lieu chinh gom MNIST, EMNIST digits va bo du lieu chu so viet tay ca nhan da duoc chuan hoa ve kich thuoc 28x28. MNIST duoc dung de huan luyen mo hinh co so, EMNIST duoc dung de mo rong kha nang tong quat hoa, con bo du lieu ca nhan duoc dung de tao ra mo hinh cuoi cung phu hop voi bai toan thuc te.

Phuong phap nghien cuu ket hop giua xay dung mo hinh hoc may va xu ly anh. Ve mat hoc may, de tai su dung mang noron tich chap CNN lam mo hinh phan lop trung tam. Ve mat tien xu ly, anh dau vao duoc dua ve dang gan voi du lieu MNIST thong qua cac buoc chuyen xam, tach nen, loc nhieu, xac dinh vung chu so, co gian va can giua. Muc tieu cua quy trinh nay la giam do lech giua anh thuc te va anh ma mo hinh da hoc, qua do cai thien chat luong nhan dang.

Trong qua trinh huan luyen, nhom ap dung chien luoc 3 giai doan. Giai doan thu nhat huan luyen mo hinh tren MNIST de hoc dac trung co ban cua chu so viet tay. Giai doan thu hai tinh chinh mo hinh tren EMNIST digits de tang kha nang thich nghi voi cac bien thien khac nhau cua du lieu. Giai doan thu ba huan luyen mo hinh tren bo du lieu ca nhan da hop nhat trong thu muc `my_digits_28`, qua do tao ra mo hinh cuoi cung dung cho suy luan va danh gia. Cach tiep can nay giup tan dung tri thuc tu bo du lieu lon, dong thoi van toi uu duoc hieu qua tren du lieu muc tieu.

De tang kha nang tong quat hoa, de tai su dung data augmentation trong qua trinh huan luyen. Cac phep quay, tinh tien va phong to/thu nho ngau nhien duoc ap dung tren anh train nham mo phong su khac biet trong cach viet tay thuc te. Ngoai ra, cac ky thuat EarlyStopping, ReduceLROnPlateau va ModelCheckpoint duoc su dung de theo doi chat luong mo hinh tren tap validation, tranh overfitting va luu lai phien ban mo hinh tot nhat. O giai doan suy luan, de tai su dung Test-Time Augmentation de tang do on dinh du doan cho cac mau kho.

Ket qua mo hinh duoc danh gia bang do chinh xac tong the, do chinh xac theo tung lop va ma tran nham lan. Ben canh cac chi so dinh luong, de tai con xuat cac mau du doan sai ra file CSV va tep anh de phan tich loi. Nhung buoc nay giup danh gia duoc ca hieu qua tong quat lan cac truong hop nham lan cu the, tu do dua ra huong cai tien hop ly cho mo hinh va du lieu.

# 3.1 Quy Trinh Xay Dung He Thong

Quy trinh xay dung he thong duoc to chuc theo chuoi buoc lien tiep gom chuan bi du lieu, tien xu ly, huan luyen mo hinh, suy luan va danh gia. Dau tien, du lieu chu so viet tay ca nhan duoc thu thap va chia thanh hai tap train va validation. Sau do, anh duoc chuyen doi ve dinh dang 28x28 thong nhat trong thu muc `my_digits_28` de phu hop voi dau vao cua mo hinh CNN. Song song voi du lieu ca nhan, he thong su dung hai bo du lieu chuan la MNIST va EMNIST digits de xay dung nen hoc dac trung.

Giai doan huan luyen duoc thuc hien qua 3 stage chinh. Stage 1 huan luyen mo hinh CNN tren MNIST de tao ra mo hinh co so. Stage 2 fine-tune mo hinh tren EMNIST digits de mo rong kha nang tong quat hoa truoc cac bien thien cua chu so viet tay. Stage 3 huan luyen mo hinh tren bo du lieu ca nhan hop nhat `my_digits_28` de tao ra mo hinh cuoi cung `stage_03_final.keras`. Toan bo quy trinh nay su dung tap validation de theo doi chat luong mo hinh va lua chon trong so tot nhat.

Khi nhan mot anh moi, he thong dua anh vao khoi tien xu ly. Tai day, anh duoc chuyen xam, nguong hoa, xu ly hinh thai hoc, loc thanh phan lien thong, cat vung chu so, co gian va can giua theo tam khoi. Anh sau tien xu ly duoc dua qua mo hinh CNN da huan luyen de thu duoc vector xac suat tren 10 lop tu 0 den 9. He thong chon nhan co xac suat cao nhat lam ket qua du doan va co the hien thi them top-k du doan de phuc vu phan tich.

Ben canh khoi suy luan, he thong con co khoi danh gia va phan tich loi. Khoi nay su dung tap validation de tinh accuracy, ma tran nham lan, xac dinh cac cap chu so hay bi nham va xuat cac mau du doan sai ra tep CSV hoac anh. Tu nhung ket qua nay, nhom co the quay lai dieu chinh du lieu, tien xu ly hoac tham so huan luyen de nang cao hieu qua he thong.

Xet tong the, he thong co 5 khoi chuc nang chinh: khoi du lieu, khoi tien xu ly, khoi huan luyen 3 stage, khoi suy luan va khoi danh gia. Cau truc nay giup he thong de quan ly, de mo rong va thuan loi khi mo ta trong bao cao do an.
