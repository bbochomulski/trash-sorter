import wx
import os
import wx.lib.statbmp as SB
from pubsub import pub
from PIL import Image
from resizeimage import resizeimage

from testing import input_photo
from keras.models import load_model


PhotoWidth = 512
PhotoHeight = 384


class Frame(wx.Frame):
    def __init__(self, parent, title):
        super(Frame, self).__init__(parent, title=title)
        self.panel = wx.Panel(self)
        self.model = load_model(os.path.join('models', 'hindus_v7', 'hindus_v7_pass6'))
        self.dragged_image = None
        st = wx.StaticText(self.panel, label="Sprawdź do jakiego pojemnika możesz\nwrzucić odpad",
                           style=wx.ALIGN_CENTER)

        font = st.GetFont()

        font.PointSize += 5
        font = font.Bold()
        st.SetFont(font)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.sizer.Add(st, wx.SizerFlags().Border(wx.LEFT, 90))
        self.panel.SetSizer(self.sizer)

        button = wx.Button(self.panel, wx.ID_ANY, 'Sprawdź gdzie wyrzucić', (170, 550))
        self.Bind(wx.EVT_BUTTON, self.onButton, button)

        button.SetBackgroundColour((255, 255, 255, 1))
        button.SetWindowStyleFlag(wx.SIMPLE_BORDER)
        button.Bind(wx.EVT_BUTTON, self.onButton)

        bitmap = wx.Bitmap("images/recycling.jpg")
        img = bitmap.ConvertToImage()
        img = img.Scale(512, 384, wx.IMAGE_QUALITY_HIGH)

        self.image_ctrl = SB.GenStaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(img))

        image_drop = DropImage(self)

        self.image_ctrl.SetDropTarget(image_drop)
        self.sizer.Add(self.image_ctrl, wx.SizerFlags().Center())
        self.panel.SetSizer(self.sizer)
        self.sizer.Add(button, wx.SizerFlags().Border(wx.LEFT, 200))
        self.panel.Refresh()
        self.panel.SetSizer(self.sizer)
        self.sizer.Fit(self)

        pub.subscribe(self.update_image_on_drag, 'imgDragged')

    def update_image_on_drag(self, filepath):
        self.on_view(filepath=filepath)

    def on_view(self, filepath):
        img = Image.open(filepath)
        self.dragged_image = img
        img = resizeimage.resize_contain(img, [PhotoWidth, PhotoHeight])
        wx_img = wx.Image(img.size[0], img.size[1])
        wx_img.SetData(img.convert('RGB').tobytes())
        bmp = wx.Bitmap(wx_img)
        self.image_ctrl.SetBitmap(bmp)
        self.image_ctrl.Refresh()

    def onButton(self, event):
        if self.dragged_image is not None:
            verdict = 'Na obrazku znajduje sie {}'.format(input_photo(self.model, self.dragged_image))
            msg = wx.MessageDialog(None, verdict, 'Wynik', wx.OK | wx.ICON_INFORMATION)
        else:
            verdict = 'Nie wybrałeś obrazka'
            msg = wx.MessageDialog(None, verdict, 'Wynik', wx.ICON_ERROR | wx.ICON_INFORMATION)

        msg.ShowModal()
        msg.Destroy()


class DropImage(wx.FileDropTarget):

    def __init__(self, widget):
        wx.FileDropTarget.__init__(self)
        self.widget = widget

    def OnDropFiles(self, x, y, filenames):
        image = Image.open(filenames[0])
        image.thumbnail((PhotoWidth, PhotoHeight))
        image.save('test/img_to_check.png')
        pub.sendMessage('imgDragged', filepath='test/img_to_check.png')
        return True


class MyApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = None

    def OnInit(self):
        self.frame = Frame(parent=None, title="Trash sorter")
        self.frame.Show()

        return True


if __name__ == '__main__':
    app = MyApp()
    app.MainLoop()
