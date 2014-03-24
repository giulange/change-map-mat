%% input

load iMap1_chunk_#0
load iMap1_chunk_#1
load iMap1_chunk_#2
load iMap1_chunk_#3
load iMap1_chunk_#4
load iMap1_chunk_#5

figure(11)
subplot(611),imshow(iMap1_chunk__0,'InitialMagnification',5),ylabel('gpu[0]')
subplot(612),imshow(iMap1_chunk__1,'InitialMagnification',5),ylabel('gpu[1]')
subplot(613),imshow(iMap1_chunk__2,'InitialMagnification',5),ylabel('gpu[0]')
subplot(614),imshow(iMap1_chunk__3,'InitialMagnification',5),ylabel('gpu[1]')
subplot(615),imshow(iMap1_chunk__4,'InitialMagnification',5),ylabel('gpu[0]')
subplot(616),imshow(iMap1_chunk__5,'InitialMagnification',5),ylabel('gpu[1]')

%% output

load oMap_chunk_#0
load oMap_chunk_#1
load oMap_chunk_#2
load oMap_chunk_#3
load oMap_chunk_#4
load oMap_chunk_#5

figure(10)
subplot(611),imshow(oMap_chunk__0,'InitialMagnification',5),ylabel('gpu[0]')
subplot(612),imshow(oMap_chunk__1,'InitialMagnification',5),ylabel('gpu[1]')
subplot(613),imshow(oMap_chunk__2,'InitialMagnification',5),ylabel('gpu[0]')
subplot(614),imshow(oMap_chunk__3,'InitialMagnification',5),ylabel('gpu[1]')
subplot(615),imshow(oMap_chunk__4,'InitialMagnification',5),ylabel('gpu[0]')
subplot(616),imshow(oMap_chunk__5,'InitialMagnification',5),ylabel('gpu[1]')

